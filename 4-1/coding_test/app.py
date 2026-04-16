import json
import subprocess
import sys
import tempfile
import os
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "AUE4023_mid.ipynb")


def load_problems():
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    problems = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        if cell["cell_type"] == "markdown":
            desc = "".join(cell["source"]).strip()
            lines = [l for l in desc.splitlines() if l.strip()]

            # 챕터 헤더(한 줄, ## 로 시작)는 건너뜀
            if len(lines) == 1 and lines[0].startswith("## "):
                i += 1
                continue

            # 바로 다음에 오는 code 셀 찾기 (연속 markdown은 설명으로 합침)
            j = i + 1
            accumulated_desc = desc
            while j < len(cells) and cells[j]["cell_type"] == "markdown":
                extra = "".join(cells[j]["source"]).strip()
                # 챕터 헤더가 나오면 중단
                extra_lines = [l for l in extra.splitlines() if l.strip()]
                if len(extra_lines) == 1 and extra_lines[0].startswith("## "):
                    break
                accumulated_desc += "\n\n" + extra
                j += 1

            if j < len(cells) and cells[j]["cell_type"] == "code":
                answer = "".join(cells[j]["source"]).strip()
                problems.append(
                    {
                        "id": len(problems),
                        "description": accumulated_desc,
                        "answer": answer,
                    }
                )
                i = j + 1
                continue
        i += 1
    return problems


PROBLEMS = load_problems()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/problems")
def get_problems():
    return jsonify(
        [{"id": p["id"], "title": _extract_title(p["description"])} for p in PROBLEMS]
    )


@app.route("/api/problem/<int:pid>")
def get_problem(pid):
    if pid < 0 or pid >= len(PROBLEMS):
        return jsonify({"error": "not found"}), 404
    p = PROBLEMS[pid]
    return jsonify({"id": p["id"], "description": p["description"]})


@app.route("/api/run", methods=["POST"])
def run_code():
    data = request.json
    code = data.get("code", "")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return jsonify(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"stdout": "", "stderr": "시간 초과 (10초)", "returncode": -1})
    finally:
        os.unlink(tmp_path)


@app.route("/api/reveal/<int:pid>")
def reveal_answer(pid):
    if pid < 0 or pid >= len(PROBLEMS):
        return jsonify({"error": "not found"}), 404
    return jsonify({"answer": PROBLEMS[pid]["answer"]})


def _extract_title(desc: str) -> str:
    for line in desc.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:60]
    return "문제"


if __name__ == "__main__":
    print(f"문제 {len(PROBLEMS)}개 로드됨")
    app.run(debug=False, port=5555)
