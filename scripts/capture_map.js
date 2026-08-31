#!/usr/bin/env node

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const count = Number(option("count", "100"));
const seed = Number(option("seed", "42"));
const port = Number(option("port", "3000"));
if (!Number.isInteger(count) || count <= 0) throw new Error("--count must be a positive integer");
if (!Number.isInteger(seed)) throw new Error("--seed must be an integer");

const bounds = { latMin: 45.46, latMax: 45.48, lonMin: 9.18, lonMax: 9.22 };
const imageDir = path.resolve("data/generated/images");
const overlayDir = path.resolve("data/generated/overlays");
const manifestPath = path.resolve("data/generated/manifest.jsonl");
fs.mkdirSync(imageDir, { recursive: true });
fs.mkdirSync(overlayDir, { recursive: true });

function mulberry32(initialSeed) {
  let state = initialSeed >>> 0;
  return function random() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const random = mulberry32(seed);
const coordinates = Array.from({ length: count }, () => ({
  lat: bounds.latMin + random() * (bounds.latMax - bounds.latMin),
  lon: bounds.lonMin + random() * (bounds.lonMax - bounds.lonMin),
}));

async function waitForMapIdle(page) {
  await page.waitForFunction(() => window.mapReady === true);
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        let timer;
        const done = () => {
          window.map.events.remove("render", rendered);
          resolve();
        };
        const rendered = () => {
          clearTimeout(timer);
          timer = setTimeout(done, 250);
        };
        window.map.events.add("render", rendered);
        rendered();
      }),
  );
}

async function capture(page, map, destination) {
  await waitForMapIdle(page);
  await map.screenshot({ path: destination, animations: "disabled" });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 600, height: 650 } });
    await page.goto(`http://localhost:${port}/web/`, { waitUntil: "networkidle" });
    await page.locator("button").evaluate((element) => {
      element.style.display = "none";
    });
    const map = page.locator("#map");
    fs.writeFileSync(manifestPath, "", "utf8");

    for (const [index, coordinate] of coordinates.entries()) {
      await page.evaluate(
        ({ lon, lat }) => window.map.setCamera({ center: [lon, lat] }),
        coordinate,
      );
      const filename = `${coordinate.lat.toFixed(5)}_${coordinate.lon.toFixed(5)}.png`;

      await page.evaluate(() => window.setMask(false));
      await capture(page, map, path.join(imageDir, filename));
      await page.evaluate(() => window.setMask(true));
      await capture(page, map, path.join(overlayDir, filename));
      fs.appendFileSync(
        manifestPath,
        `${JSON.stringify({
          id: path.parse(filename).name,
          latitude: coordinate.lat,
          longitude: coordinate.lon,
          crs: "EPSG:4326",
          zoom: 19,
          viewport: [600, 600],
          imageryProvider: "Azure Maps satellite",
          image: `images/${filename}`,
          overlay: `overlays/${filename}`,
        })}\n`,
      );
      process.stdout.write(`\rCaptured ${index + 1}/${coordinates.length}`);
    }
    process.stdout.write("\n");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
