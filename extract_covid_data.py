import requests
import lxml.html as lh
import pandas as pd

url = 'https://www.bphc.org/onlinenewsroom/Blog/Lists/Posts/Post.aspx?List=24ee0d58%2D2a85%2D4a4a%2D855b%2Df5af9d781627&ID=1282&RootFolder=%2Fonlinenewsroom%2FBlog%2FLists%2FPosts&Source=https%3A%2F%2Fwww%2Ebphc%2Eorg%2FPages%2Fdefault%2Easpx&Web=03126e14%2D4972%2D4333%2Db8a3%2D800cbc1cafce'
page = requests.get(url)
doc = lh.fromstring(page.content)
table = doc.xpath('//tbody')[-1]


for row in table.iterchildren():
    for col in row.iterchildren():
        print(col.text_content().strip())