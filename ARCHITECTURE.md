# Architecture notes

A quick map of what's in this repo and why it's organized this way.

## Two implementations, same logic

We built the core logic twice on purpose:

- **`dashboard/`** — the live demo. Pure JavaScript, runs in a browser, no install. This is what you should actually click through.
- **`src/` + `api/`** — the same flood detection, classifier, priority scoring, and routing logic, written in Python and exposed through a FastAPI service. It's not connected to the dashboard — it's there to show the logic holds up as a real backend, and it's the part our automated tests cover.

## `dashboard/resqai_dashboard.html`

One file, everything in it: layout, styling, and all the JavaScript. It calls six free public APIs directly from the browser (see the README for the list) and uses Leaflet.js for the map. There's no server involved — the browser does all the work.

## `src/flood/detector.py`

Takes an image, converts it to HSV color space, and flags water-colored pixels using a combination of hue/saturation thresholding and Otsu's method. Returns a coverage percentage and a confidence score. OpenCV + NumPy, nothing fancier — a real trained model would slot in here later without changing anything else.

## `src/nlp/classifier.py`

Keyword-based classifier for English, Urdu script, and Roman Urdu. Maps text to a category (rescue needed, medical emergency, flood, road blocked, etc.) and a severity level. No ML model — deliberately simple and fully testable.

## `src/risk/priority_engine.py`

The actual formula: 30% people risk + 25% flood severity + 20% infrastructure damage + 15% road accessibility + 10% weather risk. Returns the score plus which factors drove it, because a number with no explanation isn't useful to a responder making a call under pressure.

## `src/routing/zones_and_routes.py`

Splits an area into a grid of zones, and picks the best of several candidate routes given a set of blocked roads.

## `api/main.py`

FastAPI wrapper around the four modules above. Run it and check `/docs` for the auto-generated endpoint list.

## `tests/test_core.py`

Eight tests, one per core behavior. Run with `python3 tests/test_core.py` — no pytest install required, though it's compatible if you have it.
