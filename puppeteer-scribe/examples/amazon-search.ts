import puppeteer from "puppeteer";
import Scribe from "../index";

(async () => {
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: null,
  });
  const page = await browser.newPage();

  const scribe = new Scribe(page, {
    visualize: true,
  });

  await page.goto("https://www.amazon.com/");

  await page.waitForSelector("#twotabsearchtextbox");

  await scribe.click("#twotabsearchtextbox");

  await scribe.type("ysl myself largest most expensive bottle");

  await scribe.click("#nav-search-submit-button");

  await page.waitForNavigation();

  for (let i = 1; i <= 5; i++) {
    await scribe.click(`#a-autoid-${i}-announce`);
  }
})();
