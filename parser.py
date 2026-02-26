from typing import Any
from bs4 import BeautifulSoup, Tag
import requests
from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
from pathvalidate import sanitize_filename
import logging


logger = logging.getLogger("bird_bench_parser")

def setup_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def get_timestamp(date: str) -> int:
    formats = ("%b %d, %Y", "%B %d, %Y")
    for fmt in formats:
        try:
            ts = datetime.strptime(date, fmt).replace(tzinfo=timezone.utc).timestamp()
            return int(ts)
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date}")


def get_url(tag : Tag) -> str | None:
    link = tag.find("a")
    return (link.get("href") or None) if link else None


def get_percent(tag : Tag) -> float | None:
    txt = tag.text.strip()
    try:
        return float(txt)
    except ValueError:
        return None


def get_model(tag : Tag) -> str:
    parts = []
    for child in tag.children:
        if type(child) == Tag and ('affiliation' in (child.get("class") or []) or child.select_one("span.affiliation")):
            break
        if child.name == "br" or child == "\n":
            continue
        parts.append(child.text.strip())
    text = " ".join(parts)
    return text


def download_github_readme(url: str, file_path: Path, session: requests.Session) -> None:
    p = urlparse(url)
    owner, repo = p.path.strip("/").split("/")[:2]

    for name in ("README.md", "README.rst", "README", ".github/README.md", "docs/README.md"):
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{name}"
        r = session.get(raw_url, timeout=30)
        if r.status_code == 200:
            file_path.write_bytes(r.content)
            return
        if r.status_code != 404:
            r.raise_for_status()

    api = f"https://api.github.com/repos/{owner}/{repo}/readme"
    api_r = session.get(api, headers={"Accept": "application/vnd.github+json"}, timeout=30)
    api_r.raise_for_status()

    raw_readme_url = api_r.json()["download_url"]
    r = session.get(raw_readme_url, timeout=30)
    r.raise_for_status()

    file_path.write_bytes(r.content)


def download_hf_readme(url: str, file_path: Path) -> None:
    p = urlparse(url)
    ns, repo = p.path.strip("/").split("/")[:2]
    repo_id = f"{ns}/{repo}"

    hf_hub_download(
        repo_id=repo_id,
        filename="README.md",
        local_dir=file_path.parent
    )

    (file_path.parent / "README.md").replace(file_path)


def find_table_rows(content: BeautifulSoup) -> list[Tag]:
    tab_container = content.find("div", id="overall-leaderboard")
    if not tab_container:
        raise RuntimeError('Can\'t find "Overall Leaderboard" tab')

    for card in tab_container.find_all("div", class_="card card-outline-secondary"):
        header = card.find("div", class_="card-header")
        if header.text.strip() == "Leaderboard - Execution Accuracy (EX)":
            rows = card.find("tbody").find_all("tr")
            if len(rows) <= 1:
                raise RuntimeError('Leaderboard - Execution Accuracy (EX) table has no rows')
            return rows

    raise RuntimeError('Can\'t find "Leaderboard - Execution Accuracy (EX)" table')


def parse_row(pos: int, row: Tag) -> dict[str, Any]:
    item = {"position": pos}
    tds = row.find_all("td")
    item["submission_date"] = get_timestamp(tds[0].text.strip())
    item["model"] = get_model(tds[1])
    item["submitted_by"] = tds[1].find(class_="affiliation").text.strip()
    item["paper_url"] = get_url(tds[1])
    item["code_url"] = get_url(tds[2])
    item["paper_path"] = None
    item["code_path"] = None
    item["oracle_knowledge"] = tds[4].text.strip() == "✔️"
    item["dev"] = get_percent(tds[5])
    item["test"] = get_percent(tds[6])
    return item


def download_paper(item: dict, paper_dir: Path, session: requests.Session) -> None:
    paper_url = item.get("paper_url")
    if not paper_url:
        return

    safe = sanitize_filename(item.get("model"))
    try:
        if "arxiv.org" in paper_url:
            file_path = paper_dir / f"{item.get("submission_date")}_{safe}_paper.pdf"
            r = session.get(paper_url.replace("abs", "pdf", 1), timeout=30)
        else:
            file_path = paper_dir / f"{item.get("submission_date")}_{safe}_paper.html"
            r = session.get(paper_url, timeout=30)

        r.raise_for_status()
        file_path.write_bytes(r.content)
        item["paper_path"] = str(file_path)

    except requests.RequestException as e:
        logger.warning("[pos=%s] Paper download failed: %s", item.get("position"), e)


def download_readme(item: dict, readme_dir: Path, session: requests.Session) -> None:
    code_url = item.get("code_url")
    if not code_url:
        return

    safe = sanitize_filename(item.get("model"))
    file_path = readme_dir / f"{item.get("submission_date")}_{safe}_code.md"

    if "github" in code_url:
        try:
            download_github_readme(code_url, file_path, session)
            item["code_path"] = str(file_path)
        except requests.RequestException as e:
            logger.warning("[pos=%s] Github download failed: %s", item.get("position"), e)

    elif "huggingface" in code_url:
        try:
            download_hf_readme(code_url, file_path)
            item["code_path"] = str(file_path)
        except Exception as e:
            logger.warning("[pos=%s] HF download failed: %s", item.get("position"), e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, default=Path("bird-bench.json"))
    parser.add_argument("--no-download-papers", action="store_true")
    parser.add_argument("--papers-path", type=Path, default=Path("paper"))
    parser.add_argument("--no-download-readmes", action="store_true")
    parser.add_argument("--readmes-path", type=Path, default=Path("code"))
    args = parser.parse_args()
    setup_logging()

    url = "https://bird-bench.github.io/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = BeautifulSoup(response.text, "lxml")

    rows = find_table_rows(content)
    items = []
    for i, row in enumerate(rows[1:], start=1):
        items.append(parse_row(i, row))

    with requests.Session() as session:
        if not args.no_download_papers:
            paper_dir = args.papers_path
            paper_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                download_paper(item, paper_dir, session)

        if not args.no_download_readmes:
            readme_dir = args.readmes_path
            readme_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                download_readme(item, readme_dir, session)


    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()