from bs4 import BeautifulSoup, Tag
import requests
from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_timestamp(date: str) -> int:
    formats = ("%b %d, %Y", "%B %d, %Y")
    for fmt in formats:
        try:
            ts = datetime.strptime(date, fmt).replace(tzinfo=timezone.utc).timestamp()
            return int(ts)
        except ValueError:
            pass
    raise ValueError(f"Unknown date format: {date}")


def get_url(tag : Tag):
    link = tag.find("a")
    return (link.get("href") or None) if link else None


def get_percent(tag : Tag):
    txt = tag.text.strip()
    try:
        return float(txt)
    except ValueError:
        return None

def get_model(tag : Tag):
    parts = []
    for child in tag.children:
        if type(child) == Tag and ('affiliation' in (child.get("class") or []) or child.select_one("span.affiliation")):
            break
        if child.name == "br" or child == "\n":
            continue
        parts.append(child.text.strip())
    text = " ".join(parts)
    return text


def download_github_readme(url: str, file_path: Path):
    p = urlparse(url)
    owner, repo = p.path.strip("/").split("/")[:2]

    api = f"https://api.github.com/repos/{owner}/{repo}/readme"

    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    api_r = requests.get(api, headers=headers, timeout=30)
    api_r.raise_for_status()

    raw_readme_url = api_r.json()["download_url"]

    r = requests.get(raw_readme_url, timeout=30)
    r.raise_for_status()

    file_path.write_bytes(r.content)


def download_hf_readme(url: str, file_path: Path):
    p = urlparse(url)
    ns, repo = p.path.strip("/").split("/")[:2]
    repo_id = f"{ns}/{repo}"

    try:
        readme_cache_path = hf_hub_download(repo_id=repo_id, filename="README.md")
        file_path.write_bytes(Path(readme_cache_path).read_bytes())
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, default=Path("bird-bench.json"))
    parser.add_argument("--no-download-papers", action="store_true")
    parser.add_argument("--papers-path", type=Path, default=Path("paper"))
    parser.add_argument("--no-download-readmes", action="store_true")
    parser.add_argument("--readmes-path", type=Path, default=Path("code"))
    args = parser.parse_args()

    url = "https://bird-bench.github.io/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = BeautifulSoup(response.text, "lxml")

    rows = content.find("table").find("tbody").find_all("tr")
    items = []
    for i, row in enumerate(rows[1:], start=1):
        item = {"position": i}
        tds = row.find_all("td")
        item["submission_date"] = get_timestamp(tds[0].text.strip())
        item["model"] = get_model(tds[1])
        item["submitted_by"] = tds[1].find(class_="affiliation").text.strip()
        item["paper_url"] = get_url(tds[1])
        item["code_url"] = get_url(tds[2])
        item["oracle knowledge"] = True if tds[4].get_text() else False
        item["dev"] = get_percent(tds[5])
        item["test"] = get_percent(tds[6])
        items.append(item)

    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

    if not args.no_download_papers:
        paper_dir = Path(args.papers_path)
        paper_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            paper_url = item.get("paper_url")
            if not paper_url:
                continue

            try:
                if "arxiv.org" in paper_url:
                    file_path = paper_dir / f"{item.get("submission_date")}_{item.get("model")}_paper.pdf"
                    r = requests.get(paper_url.replace("abs", "pdf", 1), timeout=30)
                else:
                    file_path = paper_dir / f"{item.get("submission_date")}_{item.get("model")}_paper.html"
                    r = requests.get(paper_url, timeout=30)

                r.raise_for_status()
                file_path.write_bytes(r.content)

            except requests.RequestException as e:
                print(f"[pos={item.get("position")}] Paper download failed: {e}")
                continue


    if not args.no_download_readmes:
        readme_dir = Path(args.readmes_path)
        readme_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            code_url = item.get("code_url")
            if not code_url:
                continue
            file_path = readme_dir / f"{item.get("submission_date")}_{item.get("model")}_code.md"

            if "github" in code_url:
                try:
                    download_github_readme(code_url, file_path)
                except requests.HTTPError as e:
                    print(f"[pos={item.get("position")}] Github download failed: {e}")
                except requests.RequestException as e:
                    print(f"[pos={item.get("position")}] {e}")

            if "huggingface" in code_url:
                try:
                    download_hf_readme(code_url, file_path)
                except Exception as e:
                    print(f"[pos={item.get("position")}] HF download failed: {e}")



if __name__ == "__main__":
    main()