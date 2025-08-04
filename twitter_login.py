from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="./my_chrome_profile",
        headless=False
    )
    page = context.new_page()
    page.goto("https://twitter.com/search?q=%23nifty50&src=typed_query&f=live")
    input("Press Enter after login is complete and tweets are loaded...")
    context.storage_state(path="twitter_storage.json")
    context.close()
