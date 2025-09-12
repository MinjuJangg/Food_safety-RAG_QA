import asyncio
import pandas as pd
import time
import random
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlaywrightScraper:
    def __init__(self, headless=True, timeout=30000):
        self.headless = headless
        self.timeout = timeout
        self.browser = None
        self.context = None
        
    async def setup(self):
        """Initialize browser and context with stealth settings"""
        playwright = await async_playwright().start()
        
        # Launch browser with stealth settings
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-notifications',
                '--disable-popup-blocking',
                '--disable-infobars',
                '--disable-extensions',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create context with stealth settings
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
        )
        
        # Add script to remove webdriver property
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
    async def cleanup(self):
        """Close browser and context"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
            
    async def remove_popups_and_cookies(self, page):
        """Remove cookie banners, popups, and ads"""
        try:
            # Remove cookie banners and popups
            await page.evaluate("""
                // Remove cookie banners and popups
                const selectors = [
                    '[class*="cookie"]',
                    '[id*="cookie"]',
                    '[class*="popup"]',
                    '[id*="popup"]',
                    '[class*="banner"]',
                    '[id*="banner"]',
                    '[class*="consent"]',
                    '[id*="consent"]',
                    '[class*="gdpr"]',
                    '[id*="gdpr"]',
                    '[class*="modal"]',
                    '[id*="modal"]',
                    '[aria-label*="cookie" i]',
                    '[aria-label*="consent" i]',
                    '.fixed',
                    '.sticky'
                ];
                
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        const style = window.getComputedStyle(el);
                        if (style.position === 'fixed' || style.position === 'sticky') {
                            el.remove();
                        }
                    });
                });
                
                // Remove ads
                const adSelectors = [
                    'ins.adsbygoogle',
                    '[id*="google_ads"]',
                    '[id*="adsense"]',
                    '[class*="adsense"]',
                    'iframe[src*="googlesyndication"]',
                    'iframe[src*="doubleclick"]',
                    'iframe[src*="googleads"]'
                ];
                
                adSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
                
                // Click common accept/close buttons
                const buttonTexts = [
                    'accept', 'agree', 'ok', 'close', 'dismiss', 'got it',
                    'accept all', 'agree all', 'continue', 'allow all',
                    '동의', '닫기', '확인', '모두 동의', '모두 거부'
                ];
                
                buttonTexts.forEach(text => {
                    const buttons = Array.from(document.querySelectorAll('button, a, div, span')).filter(
                        el => el.textContent.toLowerCase().includes(text.toLowerCase())
                    );
                    buttons.forEach(button => {
                        if (button.offsetParent !== null) {
                            button.click();
                        }
                    });
                });
            """)
            
            # Wait a bit for changes to take effect
            await page.wait_for_timeout(1000)
            
        except Exception as e:
            logger.warning(f"Error removing popups: {e}")
            
    async def scrape_page(self, url, max_retries=3):
        """Scrape a single page with retry logic"""
        for attempt in range(max_retries):
            try:
                page = await self.context.new_page()
                
                # Set timeout
                page.set_default_timeout(self.timeout)
                
                # Navigate to page
                response = await page.goto(url, wait_until='domcontentloaded')
                
                if not response:
                    return "LOG : NO RESPONSE", 999
                    
                status_code = response.status
                
                # Skip problematic sites
                if "fooddive.com" in url or "newsweek.com" in url:
                    await page.close()
                    return "LOG : BLOCKED SITE", 101
                
                # Wait for page to load
                await page.wait_for_timeout(2000)
                
                # Remove popups and cookies
                await self.remove_popups_and_cookies(page)
                
                # Scroll to load dynamic content
                await self.scroll_page(page)
                
                # Get page content
                html_content = await page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                await page.close()
                return soup.prettify(), status_code
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(2, 5))
                else:
                    try:
                        await page.close()
                    except:
                        pass
                    return f"LOG : EXECUTION ERROR - {str(e)}", 999
                    
    async def scroll_page(self, page):
        """Scroll page to load dynamic content"""
        try:
            # Get initial scroll height
            last_height = await page.evaluate('document.body.scrollHeight')
            
            while True:
                # Scroll to bottom
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                
                # Wait for new content to load
                await page.wait_for_timeout(random.randint(1000, 2000))
                
                # Scroll up slightly
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight - 50)')
                await page.wait_for_timeout(500)
                
                # Calculate new scroll height
                new_height = await page.evaluate('document.body.scrollHeight')
                
                # Break if no new content loaded
                if new_height == last_height:
                    break
                    
                last_height = new_height
                
        except Exception as e:
            logger.warning(f"Error during scrolling: {e}")
            
    async def scrape_urls(self, urls, batch_size=10):
        """Scrape multiple URLs in batches"""
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            
            # Create tasks for concurrent scraping
            tasks = [self.scrape_page(url) for url in batch]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append((f"LOG : EXCEPTION - {str(result)}", 999))
                else:
                    results.append(result)
                    
            # Add delay between batches
            if i + batch_size < len(urls):
                await asyncio.sleep(random.uniform(5, 10))
                
        return results

async def main():
    """Main scraping function"""
    # File paths
    # data_file, output_file 경로 개인별로 수정해서 돌려주세요!!
    data_file = '/home/food/people/minju/crawling/data/preprocessed/scraped_others_non200_07.csv'
    output_file = '/home/food/people/minju/crawling/data/result/data_2024_playwright_scraped_07.csv'
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} URLs from {data_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize scraper
    scraper = PlaywrightScraper(headless=True, timeout=30000)
    await scraper.setup()
    
    try:
        # Get URLs
        urls = df['사이트'].tolist()
        
        # Scrape with progress bar
        logger.info("Starting scraping...")
        results = []
        
        # Process in smaller batches with progress tracking
        batch_size = 5
        for i in tqdm(range(0, len(urls), batch_size), desc="Scraping URLs"):
            batch_urls = urls[i:i + batch_size]
            batch_results = await scraper.scrape_urls(batch_urls, batch_size=batch_size)
            results.extend(batch_results)
            
            # Periodic save
            if i % 5 == 0 and i > 0:
                temp_df = df.copy()
                current_results = len(results)
                temp_df['html'] = [r[0] for r in results] + [''] * (len(df) - current_results)
                temp_df['response'] = [r[1] for r in results] + [0] * (len(df) - current_results)
                temp_df.to_csv(output_file.replace('.csv', '_temp.csv'), 
                             encoding='utf-8-sig', index=False)
                logger.info(f"Saved progress: {current_results} URLs processed")
        
        # Save final results
        if len(results) == len(df):
            df['html'] = [r[0] for r in results]
            df['response'] = [r[1] for r in results]
        else:
            logger.warning(f"Result count mismatch: {len(results)} vs {len(df)}")
            # Pad results to match dataframe length
            df['html'] = [r[0] for r in results] + [''] * (len(df) - len(results))
            df['response'] = [r[1] for r in results] + [0] * (len(df) - len(results))
        df.to_csv(output_file, encoding='utf-8-sig', index=False)
        
        logger.info(f"Scraping completed. Results saved to {output_file}")
        
        # Print statistics
        success_count = sum(1 for r in results if not str(r[0]).startswith('LOG :'))
        logger.info(f"Successfully scraped: {success_count}/{len(results)} URLs")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    # Fix Windows asyncio issues
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())