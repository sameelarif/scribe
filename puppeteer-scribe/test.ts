import puppeteer from "puppeteer";
import Scribe from "./index";

(async () => {
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: null,
  });
  const page = await browser.newPage();

  const scribe = new Scribe(page, {
    visualize: true,
  });

  // Set the viewport size to match the screen dimensions
  await page.setViewport({ width: 1920, height: 1080 });

  await page.goto("https://www.google.com");

  await scribe.click('textarea[title="Search"]');

  await scribe.type("how long are dolphins in feet");

  await scribe.click('input[aria-label="Google Search"]');
})();
