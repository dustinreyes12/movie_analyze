import scrapy
import pickle

class IMDBSpider(scrapy.Spider):
    name = 'imdb_spider'

    # Use a delay on the scraping to avoid overloading IMDB's servers!
    custom_settings = {
        "DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 8,
        "HTTPCACHE_ENABLED": True
    }

    # Load the list of title IDs from the pickle file
    # created in the previous section
    with open("data2.pkl", 'rb') as picklefile:
        links = list(pickle.load(picklefile))

    # Generate the list of urls to scrape (based on title ID)
    start_urls = [
#         'https://www.imdb.com/title/tt0974015/'
        'http://www.imdb.com/title/%s/' % l for l in links
    ]

    # Methods go here!
    def parse(self, response):
        # Extract the links to the individual festival pages
        if 'Box Office' in response.xpath('//*[@id="titleDetails"]/h3[1]/text()').extract():
            title_id = response.url.split('/')[-2]
            title = response.xpath(
                '//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[2]/div[2]/h1/text()').extract()[0].replace('\xa0', '')
            release = response.xpath(
                '//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[2]/div[2]/div/a[4]/text()').extract()[0].replace('\n','')
            
            try:
                rating = response.xpath('//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[2]/div[2]/div/text()[1]').extract()[0].replace('\n','').strip()
            except:
                rating = ''
            
            
            
            try:
                director = response.xpath(
                    '//*[@id="title-overview-widget"]/div[2]/div[1]/div[2]/a/text()').extract()[0]
            except:
                director = ''

            moneys = response.xpath(
                '//h3[@class="subheading"]')[0].xpath('following-sibling::div/text()').re(r'\$[0-9,]+')
            money_labels = response.xpath(
                '//h3[@class="subheading"]')[0].xpath('following-sibling::div/h4/text()').extract()
            moneys = [i.replace(',', '').replace('$', '') for i in moneys]

            budget = ''
            opening = ''
            gross = ''
            worldwide_gross = ''
            try:
                for m, l in zip(moneys, money_labels[:len(moneys)]):
                    if 'budget' in l.lower():
                        budget = m
                    elif 'opening' in l.lower():
                        opening = m
                    elif 'worldwide' in l.lower():
                        worldwide_gross = m
                    elif 'gross' in l.lower():
                        gross = m
                    else:
                        continue
            except:
                pass

            try:
                metacritic_score = response.xpath(
                    '//div[@class="titleReviewBarItem"]/a/div/span/text()').extract()[0]
            except:
                metacritic_score = ''

            yield {
                'title_id': title_id,
                'title': title,
                'release': release,
                'director': director,
                'budget': budget,
                'opening': opening,
                'gross': gross,
                'worldwide_gross': worldwide_gross,
                'metacritic_score': metacritic_score,
                'mpaa_rating': rating
            }





