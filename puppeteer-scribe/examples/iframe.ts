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

  await page.setViewport({ width: 1920, height: 1080 });

  await page.goto(
    "https://hcaptcha.projecttac.com/?sitekey=27a14814-d592-444c-a711-4447baf41f48"
  );

  await new Promise((resolve) => setTimeout(resolve, 5000));

  await page.waitForSelector("iframe");

  const frames = await page.frames();

  let hcaptchaFrame;

  for (const frame of frames) {
    if (frame.url().includes("hcaptcha.html#frame=checkbox")) {
      hcaptchaFrame = frame;
      break;
    }
  }

  if (hcaptchaFrame) {
    await hcaptchaFrame.waitForSelector("#checkbox");
    await scribe.click("#checkbox", hcaptchaFrame);
  } else {
    throw new Error("hcaptchaFrame is undefined");
  }
})();
