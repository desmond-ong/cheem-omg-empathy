from bs4 import BeautifulSoup
import re

link = '/Users/sonnguyen/Desktop/tmp.html'

with open(link, 'r') as reader:
    content = reader.read()



soup = BeautifulSoup(content, "html.parser")
tmp = set()
for ol in soup.find_all("ol", {"id": "vm-playlist-video-list-ol"}):
    for tag in ol.find_all('li'):
        for div in soup.find_all("div", {"class": "vm-video-item"}):
            tmp.add(div["data-video-id"])

for s in tmp:
    print("https://www.youtube.com/timedtext_editor?v={}&lang=en&name=&kind=asr&contributor_id=0&bl=vmp&action_view_track=1&ar=2#\n".format(s))