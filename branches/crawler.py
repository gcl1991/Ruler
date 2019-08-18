# -*- coding:utf-8 -*-
from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import GoogleImageCrawler
"""
parser_threads：解析器线程数目，最大为cpu数目
downloader_threads：下载线程数目，最大为cpu数目
storage：存储地址，使用字典格式。key为root_dir
keyword:浏览器搜索框输入的关键词
max_num:最大下载图片数目
"""
save_path = "/home/gcl/data/ruler/crawler"
max_down_img = 10000
key_word = "直尺"
#谷歌图片爬虫
# google_storage = {'root_dir': save_path+'/google'}
# google_crawler = GoogleImageCrawler(parser_threads=4,
#                                    downloader_threads=4,
#                                    storage=google_storage)
# google_crawler.crawl(keyword=key_word,
#                      max_num=max_down_img)


# #必应图片爬虫
bing_storage = {'root_dir': save_path+'/bing'}
bing_crawler = BingImageCrawler(parser_threads=4,
                                downloader_threads=4,
                                storage=bing_storage)
bing_crawler.crawl(keyword=key_word,
                   max_num=max_down_img)
#
#
# #百度图片爬虫
# baidu_storage = {'root_dir': '/Users/suosuo/Desktop/icrawler学习/baidu'}
#
# baidu_crawler = BaiduImageCrawler(parser_threads=4,
#                                   downloader_threads=4,
#                                   storage=baidu_storage)
# baidu_crawler.crawl(keyword='美女',
#                     max_num=10)