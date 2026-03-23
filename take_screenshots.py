import os
from playwright.sync_api import sync_playwright

def take_screenshots():
    os.makedirs('screenshots', exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Capturing Home Page...")
        page.goto('http://127.0.0.1:8000/')
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)  # Wait for any animations or image loads
        page.screenshot(path='screenshots/home.png', full_page=True)
        
        print("Capturing Recommendation Page...")
        page.goto('http://127.0.0.1:8000/swipe')
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path='screenshots/recommendation.png', full_page=True)
        
        browser.close()
        print("Screenshots saved successfully.")

if __name__ == '__main__':
    take_screenshots()
