import os
from io import BytesIO
from multiprocessing.dummy import Pool as ThreadPool
from os import getenv

import cv2
import requests
from PIL import Image
from requests import exceptions

import face

BING_API_KEY = getenv('BING_API_KEY', '')
print("API KEY: ", BING_API_KEY)
MAX_RESULTS = 150

URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
HEADERS = {"Ocp-Apim-Subscription-Key" : str(BING_API_KEY)}

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])


def get_photos(famous_people_file):
	people = []
	with open(famous_people_file) as fp:
		people = fp.read().splitlines() 
	unique_people = set(people)
	pool = ThreadPool(100) 
	pool.map(get_urls, unique_people)
	pool.close() 
	pool.join() 


def get_urls(person):
	params = {"q": person, "count": MAX_RESULTS, "imageType": "photo"}
	print("[INFO] searching Bing API for '{}'".format(person))
	search = requests.get(URL, headers=HEADERS, params=params)
	search.raise_for_status()
	results = search.json()
	# num_results = min(results["totalEstimatedMatches"], MAX_RESULTS)
	# print("[INFO] {} total results for '{}'".format(num_results, person))

	thumbnail_urls = [img["thumbnailUrl"] + '&c=7&w=250&h=250' for img in results["value"]]
	urls = pare_matches(thumbnail_urls)
	download_urls(person, urls)
	return urls


def pare_matches(thumbnail_urls):
	urls = []
	identifier = face.Identifier()
	if len(thumbnail_urls) > 1:
		for image_url in thumbnail_urls:
			match = identifier.compare_faces(thumbnail_urls[0], image_url, True)
			if match.is_match:
				urls.append(image_url)
	return urls


def download_urls(person, urls):
	if len(urls) > 5:
		directory = '../images/' + person
		if not os.path.exists(directory):
			os.makedirs(directory)
		count = 0
		for url in urls:
			image_data = requests.get(url)
			image_data.raise_for_status()
			image = Image.open(BytesIO(image_data.content))
			image_path = os.path.join(directory, str(count))
			image.save(image_path + '.jpg')
			count += 1


if __name__ == "__main__":
	get_photos('famous_people.txt')
