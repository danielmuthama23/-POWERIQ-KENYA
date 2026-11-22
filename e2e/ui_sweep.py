"""End-to-end UI demo: visits all 11 pages and exercises storage, copilot, chief assessment, approvals, audit verify and a scenario.
Requires: pip install playwright, a Chrome binary, and the app running (see README). Exits non-zero on any page error."""
from playwright.sync_api import sync_playwright
import os, sys
OUT = os.environ.get("E2E_OUT", "e2e/out"); os.makedirs(OUT, exist_ok=True)
URL = os.environ.get("POWERIQ_URL", "http://localhost:8000/")
CHROME = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")
pages = ["National Overview","Generation Intelligence","Predictive Maintenance","Grid Intelligence","Renewable Intelligence","Storage Optimization","Energy Loss Detection","Climate Intelligence","National Digital Twin","AI Command Centre","Reports and Audit","Kenya Problems & AI Analysis","Reserve Risk (Monte Carlo)","Curtailment to Storage","Drought & Wind Stress Tests","Power Station Fleet","Theft Inspection Planner","Query Studio (Ask Anything)","Fault Hub (Live Map)","Demand Structure (STL)"]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox","--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errs.append(f"console.error: {m.text}") if m.type == "error" else None)
    pg.goto(URL); pg.fill("#key", "demo-approver-key"); pg.click("text=Enter command centre")
    pg.wait_for_selector("text=National Overview")
    for name in pages:
        pg.click(f"nav button:has-text('{name}')"); pg.wait_for_timeout(2500)
        alerts = pg.locator("[role=alert]").all_inner_texts()
        print(f"{name:26s} alerts={alerts[:1]}")
    # interactions
    pg.click("nav button:has-text('Storage Optimization')"); pg.click("button:has-text('Optimise')"); pg.wait_for_timeout(5000)
    print("storage result:", pg.locator("text=Peak after").count() > 0)
    pg.screenshot(path=f"{OUT}/storage.png")
    pg.click("nav button:has-text('AI Command Centre')"); pg.click("button:has-text('What happens if geothermal')"); pg.wait_for_selector("text=Model predictions", timeout=60000); pg.wait_for_timeout(500)
    pg.click("button:has-text('Run assessment')"); pg.wait_for_selector("text=Overall risk level", timeout=90000)
    pg.screenshot(path=f"{OUT}/command.png", full_page=False)
    pg.click("nav button:has-text('Reports and Audit')"); pg.wait_for_timeout(2000)
    pg.click("button:has-text('Approve') >> nth=0"); pg.wait_for_timeout(1500); pg.click("button:has-text('Verify chain')"); pg.wait_for_selector("text=chain valid", timeout=15000)
    pg.screenshot(path=f"{OUT}/audit.png")
    pg.click("nav button:has-text('National Digital Twin')"); pg.wait_for_selector("canvas"); pg.wait_for_timeout(4000)
    pg.click("button:has-text('Run')"); pg.wait_for_selector("text=24 h metric", timeout=30000)
    pg.screenshot(path=f"{OUT}/twin_full.png", full_page=True)
    real = [e for e in errs if "GL Driver" not in e]
    print("errors:", real[:6])
    b.close()
    sys.exit(1 if real else 0)
