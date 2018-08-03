import collections
import itertools
import os
from functools import wraps
from io import BytesIO
from multiprocessing.dummy import Pool as ThreadPool
from os import getenv
from time import time

import cv2
import dhash
import pybktree
import requests
from PIL import Image
from requests import exceptions

import face

BING_API_KEY = getenv('BING_API_KEY', '')
NUM_THREADS = getenv('NUM_THREADS', 4)
MAX_RESULTS = 150

URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
HEADERS = {"Ocp-Apim-Subscription-Key" : str(BING_API_KEY)}

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

HASH_URL = collections.namedtuple('HASH_URL', 'bits url')


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper


def url_to_image(url):
	image_data = requests.get(url)
	image_data.raise_for_status()
	image = Image.open(BytesIO(image_data.content))
	return image


def url_to_img_hash(url):
	image = url_to_image(url)
	image_hash = dhash.dhash_int(image)
	return image_hash


def image_distance(x, y):
	return pybktree.hamming_distance(x.bits, y.bits)


def get_photos(famous_people_file):
	people = []
	with open(famous_people_file) as fp:
		people = fp.read().splitlines() 
	unique_people = set(people)
	pool = ThreadPool(int(NUM_THREADS)) 
	pool.map(get_urls, unique_people)
	pool.close() 
	pool.join() 


def get_urls(person):
	params = {"q": person, "count": MAX_RESULTS, "imageType": "photo"}
	print("[INFO] searching Bing API for '{}'".format(person))
	search = requests.get(URL, headers=HEADERS, params=params)
	search.raise_for_status()
	results = search.json()
	thumbnail_urls = [img["thumbnailUrl"] + '&c=7&w=250&h=250' for img in results["value"]]
	urls = pare_matches(thumbnail_urls)
	download_urls(person, urls)


@timing
def pare_matches(thumbnail_urls):
	urls = []
	# Make sure all the matches are of the same person
	identifier = face.Identifier()
	if len(thumbnail_urls) > 1:
		for image_url in thumbnail_urls:
			match = identifier.compare_faces(thumbnail_urls[0], image_url, True)
			if match.is_match:
				urls.append(image_url)
	
	# Make sure there are no duplicate images
	image_hashes = [HASH_URL(url_to_img_hash(url), url) for url in urls]
	tree = pybktree.BKTree(image_distance, image_hashes)

	to_discard = []
	urls_to_keep = []
	for image_hash in image_hashes:
		if image_hash not in to_discard:
			matches = tree.find(image_hash, 3)
			for match in matches:
				if match[1].url != image_hash.url:
					to_discard.append(match[1])
			urls_to_keep.append(image_hash.url)
	return urls_to_keep


def download_urls(person, urls):
	if len(urls) > 5:
		directory = '../images/' + person
		if not os.path.exists(directory):
			os.makedirs(directory)
		count = 0
		for url in urls:
			image = url_to_image(url)
			image_path = os.path.join(directory, str(count))
			image.save(image_path + '.jpg')
			count += 1


if __name__ == "__main__":
	get_photos('famous_people.txt')
