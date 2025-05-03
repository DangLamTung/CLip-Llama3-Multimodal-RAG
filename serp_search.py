import os
import sys
import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Adjust paths as needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

async def main():
    async with AsyncWebCrawler() as crawler:
        # Request both PDF and screenshot
        result = await crawler.arun(
            url='https://www.traveloka.com/vi-vn/explore/destination/kinh-nghiem-du-lich-vinh-hy/156830',
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                pdf=True,
                screenshot=True
            )
        )
        
        if result.success:
            # Save screenshot
            if result.screenshot:
                from base64 import b64decode
                with open(os.path.join(__location__, "screenshot.png"), "wb") as f:
                    f.write(b64decode(result.screenshot))
            
            # Save PDF
            if result.pdf:
                with open(os.path.join(__location__, "page.pdf"), "wb") as f:
                    f.write(result.pdf)

if __name__ == "__main__":
    asyncio.run(main())