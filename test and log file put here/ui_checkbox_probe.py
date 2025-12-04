from __future__ import annotations

import json

from playwright.sync_api import sync_playwright

URL = "http://localhost:8504"
DATA_FILE = r"c:\\Files\\Code\\shap_streamlit\\test and log file put here\\Advertising_Data.csv"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle")
        upload_input = page.locator('input[type="file"]')
        upload_input.set_input_files(DATA_FILE)
        page.wait_for_timeout(4000)
        page.get_by_role("radio", name="模型训练 & 评估").click()
        page.wait_for_timeout(2000)
        checkbox_info = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('div[data-testid="stCheckbox"] label')).map(label => {
                const rect = label.getBoundingClientRect();
                const text = label.querySelector('p')?.innerText.trim();
                const iconRect = label.querySelector('svg')?.getBoundingClientRect();
                return {
                    text,
                    width: rect.width,
                    height: rect.height,
                    iconWidth: iconRect ? iconRect.width : null,
                    iconHeight: iconRect ? iconRect.height : null
                };
            })
            """
        )
        print(json.dumps(checkbox_info, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
