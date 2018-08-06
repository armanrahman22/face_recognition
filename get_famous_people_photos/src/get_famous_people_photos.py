import collections
import itertools
import os
import pickle
import time
from ctypes import c_int
from functools import wraps
from io import BytesIO
from itertools import islice
from multiprocessing import Lock, Manager, Pool, Queue, Value
from multiprocessing.dummy import Pool as ThreadPool
from os import getenv
from time import time

import cv2
import dhash
import progressbar as pb
import pybktree
import requests
import tensorflow as tf
from pathos.multiprocessing import ProcessPool
from PIL import Image
from requests import exceptions

import face

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
BING_API_KEY = str(getenv('BING_API_KEY', 'c8e183c6cf57419bb0c0ee885b76bbe5'))
NUM_THREADS = int(getenv('NUM_THREADS', 50))
NUM_PROCESSES = min(int(getenv('NUM_PROCESSES', 4)), os.cpu_count())
MAX_RESULTS = 150

URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
HEADERS = {"Ocp-Apim-Subscription-Key" : BING_API_KEY}

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

HASH_URL = collections.namedtuple('HASH_URL', 'bits url')

counter = Value(c_int)  # defaults to 0
counter_lock = Lock()

widgets_urls = ['Fetching urls: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
widgets_pare = ['Pare and Download urls: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
timer = None


def increment():
	with counter_lock:
		counter.value += 1


def reset():
	global timer
	with counter_lock:
		counter.value == 0
	timer = None


def url_to_image(url):
	try:
		image_data = requests.get(url)
		image_data.raise_for_status()
		image = Image.open(BytesIO(image_data.content))
	except Exception as e:
		print(e)
		image = None
		time.sleep(120)
	return image


def url_to_img_hash(url):
	try:
		image = url_to_image(url)
		image_hash = dhash.dhash_int(image)
	except Exception as e:
		print(e)
		image_hash = None
	return image_hash


def image_distance(x, y):
	return pybktree.hamming_distance(x.bits, y.bits)


def get_photos(famous_people_file):
	global timer
	people = []
	with open(famous_people_file) as fp:
		people = fp.read().splitlines() 
	unique_people = set(people)
	total_count = len(unique_people)

	# Check if we have cached our image urls
	url_file = '../../common/text_files/image_urls.data'
	if not os.path.isfile(url_file):
		timer = pb.ProgressBar(widgets=widgets_urls, maxval=total_count).start()
		threaded_result = urls_thread(unique_people)
		urls_and_people = [_ for _ in threaded_result]
		with open(url_file, 'wb') as filehandle:  
			pickle.dump(urls_and_people, filehandle)
	else:
		with open(url_file, 'rb') as filehandle:  
			# read the data as binary data stream
			urls_and_people = pickle.load(filehandle)
	reset()
	
	timer = pb.ProgressBar(widgets=widgets_pare, maxval=total_count).start()
	pare_multi_process(urls_and_people)


def urls_thread(unique_people):
	print("[INFO] Fetching image urls with {} threads".format(NUM_THREADS))
	thread_pool = ThreadPool(NUM_THREADS) 
	urls_and_people = thread_pool.imap(safe_get_urls, unique_people)
	thread_pool.close() 
	thread_pool.join() 
	print("[INFO] Done fetching image urls")
	return urls_and_people


def pare_multi_process(urls_and_people):
	print("[INFO] Paring and downlaoding all image urls with {} processes".format(NUM_PROCESSES))
	urls, person = zip(*urls_and_people)
	pare_pool = ProcessPool(NUM_PROCESSES)
	pare_pool.imap(safe_pare_matches_and_download, urls, person)
	pare_pool.close() 
	pare_pool.join()
	print("[INFO] Done paring and downlaoding all image urls")


def safe_get_urls(*args, **kwargs):
	try:
		return get_urls(*args, **kwargs)
	except Exception as e:
		print(e)


def get_urls(person):
	params = {"q": person, "count": MAX_RESULTS, "imageType": "photo"}
	search = requests.get(URL, headers=HEADERS, params=params)
	search.raise_for_status()
	results = search.json()
	thumbnail_urls = [img["thumbnailUrl"] + '&c=7&w=250&h=250' for img in results["value"]]
	try:
		# update counter 
		increment()
		timer.update(int(counter.value))
	except Exception as e:
		print(e)
	return thumbnail_urls, person


def safe_pare_matches_and_download(*args, **kwargs):
	try:
		return pare_matches_and_download(*args, **kwargs)
	except Exception as e:
		print(e)


def pare_matches_and_download(thumbnail_urls, person):
	urls = set()
	directory = '../../common/images/' + person
	if not os.path.exists(directory) and len(thumbnail_urls) > 10:
		# Make sure all the matches are of the same person
		try:
			identifier = face.Identifier(threshold=1.0)
			images = map(identifier.download_image, thumbnail_urls)
			urls_and_embeddings = identifier.detect_encode_all(images, thumbnail_urls, True)
			anchor_embedding = urls_and_embeddings[0].embedding
			# Assume first image is of the right person and check other images are of the same person
			for other in urls_and_embeddings:
				is_match, distance = identifier.compare_embedding(anchor_embedding, other.embedding)
				# print('dist: {} between {} and {}'.format(distance,urls_and_embeddings[0].url, other.url))
				if is_match:
					urls.add(other.url)
			del identifier
		except Exception as e:
			print(e)
		
		# Make sure there are no duplicate images
		image_hashes = [HASH_URL(url_to_img_hash(url), url) for url in urls]
		tree = pybktree.BKTree(image_distance, image_hashes)
		# this makes images saved in order of similarity so we can spot duplicates easier
		sorted_image_hashes = sorted(tree)
		to_discard = []
		urls_to_keep = set()
		for image_hash in sorted_image_hashes:
			if image_hash not in to_discard:
				# gets pictures within a hamming distance of 3
				matches = tree.find(image_hash, 3)
				for match in matches:
					if match[1].url != image_hash.url:
						to_discard.append(match[1])
				urls_to_keep.add(image_hash.url)
		
		# Download the images 
		download_urls(person, list(urls_to_keep))

	# Update counter 
	try:
		increment()
		timer.update(int(counter.value))
	except Exception as e:
		print(e)


def download_urls(person, urls):
	if len(urls) > 5:
		try:
			directory = '../../common/images/' + person
			if not os.path.exists(directory):
				os.makedirs(directory)
			count = 0
			for url in urls:
				image = url_to_image(url)
				image_path = os.path.join(directory, str(count))
				image.save(image_path + '.jpg')
				count += 1
		except Exception as e:
			print(e)


if __name__ == "__main__":
	get_photos('../../common/text_files/famous_people.txt')
