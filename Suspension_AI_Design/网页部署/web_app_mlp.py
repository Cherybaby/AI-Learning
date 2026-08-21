#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
悬架运动学代理模型 · 测试集「预测 vs 真值」对照网页（Python 标准库 http.server，无需 Flask）。

与 web_app.py 的区别：web_app.py 是「输入硬点 -> 实时推理」的交互演示；
本脚本(web_app2.py)是「浏览测试集里各个真实样本，对照模型预测与 Adams 真值」的评估页面，
支持拖动滑块切换不同方案(样本)，2x4 曲线网格 + 逐样本 MAE 表。

数据来源：export_test_samples.py 导出的 nn_out/test_samples.json（预先算好，网页只读不重新推理）。
若该文件不存在，启动时会自动导出一次（默认导出全部测试集样本）。

功能：
  - GET  /                    返回对照页面 web/compare.html
  - GET  /api/test_samples    返回 nn_out/test_samples.json 的全部内容
  - GET  /api/metrics         返回 metrics.json 的 test_curve_metrics(逐量MAE/Median/P90/
                              MaxError/D2) + results_worst_samples.json(各量最差Top-N曲线)

启动：
  cd /workspace/Suspension_AI_Design
  PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app2.py            # 默认 127.0.0.1:5174
  PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app2.py --port 8080
自测（线程内启动 + 请求，不常驻）：
  PYTHONPATH=/workspace/.pylibs TORCH_THREADS=4 python3 web_app2.py --self-test

安全说明：本服务默认仅监听 127.0.0.1（本机回环），无鉴权，仅供本地演示 / 单机内网使用；
勿在未加访问控制的情况下绑定 0.0.0.0 暴露到公网。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------ #
# 路径与运行环境常量
# ------------------------------------------------------------------ #
# _HERE：本脚本(web_app.py)所在目录的绝对路径，作为其余相对路径的锚点。
_HERE = os.path.dirname(os.path.abspath(__file__))
# _LIBS：第三方 Python 依赖(如 torch/numpy)的额外搜索目录。
#        优先读环境变量 PPTX_LIBS，缺省回退到 /workspace/.pylibs。
#        若该目录存在且尚未在 sys.path 中，则插入到最前，保证优先加载。
_LIBS = os.environ.get("PPTX_LIBS") or "/workspace/.pylibs"
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# _TRAIN_DIR：同级的「训练推理」目录，存放模型产物与数据导出脚本。
#             取本目录的父目录再拼「训练推理」，故与「网页部署」并列。
_TRAIN_DIR = os.path.join(os.path.dirname(_HERE), "训练推理/MLP")

# WEB_DIR / COMPARE_PATH：前端静态资源目录与对照页面(compare.html)的完整路径。
WEB_DIR = os.path.join(_HERE, "web")
COMPARE_PATH = os.path.join(WEB_DIR, "compare.html")
# SAMPLES_PATH：预先导出的测试集样本(预测+真值+MAE)，由 export_test_samples.py 生成。
SAMPLES_PATH = os.path.join(_TRAIN_DIR, "nn_out", "test_samples.json")
# METRICS_JSON_PATH：整体评估指标，其中 test_curve_metrics 记录逐量 MAE/Median/P90/MaxError/D2。
METRICS_JSON_PATH = os.path.join(_TRAIN_DIR, "nn_out", "metrics.json")
# WORST_SAMPLES_PATH：各运动学量误差最差的 Top-N 曲线，由 visualize_results.py 生成。
WORST_SAMPLES_PATH = os.path.join(_TRAIN_DIR, "nn_out", "results_worst_samples.json")

# _LOCK / _CACHE：进程内单例缓存。
#   _CACHE 缓存已加载的 test_samples.json，避免每次请求都读盘；
#   _LOCK 在多线程(ThreadingHTTPServer)下保护首次「按需导出+读盘」不被并发重入。
_LOCK = threading.Lock()
_CACHE = None


def _ensure_samples():
    """确保测试集样本(test_samples.json)已就绪，并返回其解析后的 Python 对象。

    行为：
      1. 若进程内缓存 _CACHE 已填充，直接返回缓存(命中即不再读盘)。
      2. 若磁盘上 SAMPLES_PATH 不存在，则以子进程方式调用「训练推理」目录下的
         export_test_samples.py（--max-samples 0 表示导出全部测试集样本）生成一次；
         导出失败(返回码非 0)时抛 RuntimeError，携带子进程 stdout/stderr 便于排查。
      3. 读取 JSON 写入 _CACHE 并返回。

    返回：test_samples.json 反序列化得到的 dict（含 samples / target_names 等字段）。

    注意：本函数不是线程安全的自持锁函数，并发首次导出的互斥由调用方(do_GET)持 _LOCK 保证。
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not os.path.isfile(SAMPLES_PATH):
        print(f"[web2] 未找到 {SAMPLES_PATH}，正在调用 export_test_samples.py 生成…", flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join(_TRAIN_DIR, "export_test_samples.py"),
             "--max-samples", "0"],
            cwd=_TRAIN_DIR, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": _LIBS})
        if r.returncode != 0:
            raise RuntimeError(f"导出测试样本失败:\n{r.stdout}\n{r.stderr}")
        print(r.stdout, flush=True)
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        _CACHE = json.load(f)
    return _CACHE


def _load_curve_metrics():
    """读取「模型精度详情」所需的两份统计数据，组装成前端可直接消费的字典。

    数据来源：
      - metrics.json 的 test_curve_metrics：逐运动学量的误差统计
        (MAE / Median / P90 / MaxError / D2 等)。
      - results_worst_samples.json：每个量误差最差的 Top-N 曲线，
        由 visualize_results.py 生成。

    容错：两个文件任一缺失时，对应字段返回 None，前端据此显示"未生成"
    提示而不是报错(不会因文件缺失导致 500)。

    返回：{"curve_metrics": <dict|None>, "worst_samples": <obj|None>}。
    """
    curve_metrics = None
    if os.path.isfile(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, encoding="utf-8") as f:
            curve_metrics = json.load(f).get("test_curve_metrics")
    worst_samples = None
    if os.path.isfile(WORST_SAMPLES_PATH):
        with open(WORST_SAMPLES_PATH, encoding="utf-8") as f:
            worst_samples = json.load(f)
    return {"curve_metrics": curve_metrics, "worst_samples": worst_samples}


# ------------------------------------------------------------------ #
# HTTP 处理
# ------------------------------------------------------------------ #

class Handler(BaseHTTPRequestHandler):
    """标准库 http.server 的请求处理器，负责路由本演示网页的所有 GET 请求。

    仅实现 GET（本服务为只读演示，无任何写/改接口）。路由见 do_GET。
    server_version 会体现在响应的 Server 头中，便于识别。
    """

    server_version = "SuspensionCompareUI/1.0"

    def _send_json(self, obj, code=200):
        """把 Python 对象序列化为 UTF-8 JSON 并作为 HTTP 响应发送。

        参数：
          obj  —— 可被 json.dumps 序列化的对象。
          code —— HTTP 状态码，默认 200(成功)，错误时由调用方传 404/500 等。
        说明：ensure_ascii=False 保留中文原文；同时写入 Content-Type 与 Content-Length 头。
        """
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        """以二进制方式读取本地文件 path 并按给定 Content-Type 返回。

        参数：
          path  —— 待返回文件的绝对路径(如 compare.html)。
          ctype —— 响应的 Content-Type 头。
        容错：文件不存在时返回 404 JSON 错误，而非抛异常导致连接中断。
        """
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self._send_json({"error": f"文件不存在: {path}"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """GET 路由分发。各端点说明如下：

          GET /  |  /index.html  |  /compare.html
              返回前端对照页面 compare.html(text/html)。

          GET /api/test_samples
              返回 test_samples.json 全部内容(application/json)。
              结构：{samples:[{stroke, true, pred, mae, ...}], target_names:[...]}
              持 _LOCK 调用 _ensure_samples()，首次访问会按需触发一次样本导出。
              异常时返回 500 + {"error": <msg>}。

          GET /api/metrics
              返回「模型精度详情」数据(application/json)。
              结构：{curve_metrics:<逐量统计|None>, worst_samples:<各量最差曲线|None>}。
              异常时返回 500 + {"error": <msg>}。

          GET /api/health
              健康检查探针，恒返回 {"ok": True}。

          其它路径
              返回 404 + {"error": "not found"}。
        """
        # 去掉查询串(?后部分)，只按路径部分做精确匹配路由。
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/compare.html"):
            self._send_file(COMPARE_PATH, "text/html; charset=utf-8")
        elif path == "/api/test_samples":
            try:
                # _LOCK 保证并发首次访问时「按需导出+读盘」只执行一次。
                with _LOCK:
                    data = _ensure_samples()
                self._send_json(data)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/metrics":
            try:
                self._send_json(_load_curve_metrics())
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/health":
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):        # 降噪：只在 stderr 简单打印
        """覆盖默认访问日志，加 [web2] 前缀写到 stderr，减少标准输出噪声。"""
        sys.stderr.write("[web2] " + (fmt % args) + "\n")


def serve(host="127.0.0.1", port=5174):
    """启动常驻 HTTP 服务并阻塞运行，直到 Ctrl+C。

    参数：
      host —— 监听地址，默认 127.0.0.1(仅本机回环)。
      port —— 监听端口，默认 5174。
    流程：先 _ensure_samples() 预热样本(启动即导出/加载，避免首个请求卡顿)，
    再用 ThreadingHTTPServer(每请求一线程)提供服务；收到 KeyboardInterrupt 优雅退出。

    安全：默认仅监听 127.0.0.1、无鉴权，仅供本地演示；绑定 0.0.0.0 暴露公网需自行加访问控制。
    """
    _ensure_samples()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"预测vs真值对照网页已启动: http://{host}:{port}  (Ctrl+C 停止)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


# ------------------------------------------------------------------ #
# 自测：线程内启动服务并请求，不常驻
# ------------------------------------------------------------------ #

def self_test(port=8352):
    """在后台线程内启动服务、发起真实 HTTP 请求做端到端冒烟自测，然后关闭(不常驻)。

    参数：port —— 自测使用的本地端口，默认 8352。
    校验点：
      1. GET /                 返回 HTML，且包含"预测"与"真值"字样。
      2. GET /api/test_samples 至少含 1 个样本，首样本含 stroke / true.toe /
         pred.toe / mae.toe 等关键字段。
    返回：全部断言通过返回 True，否则 False(供 main 决定进程退出码)。
    """
    import urllib.request

    _ensure_samples()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    ok = True
    try:
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            html = r.read().decode("utf-8")
        assert "预测" in html and "真值" in html
        print(f"[self-test] GET /                -> {len(html)} bytes HTML  ✅")

        with urllib.request.urlopen(base + "/api/test_samples", timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        assert len(data["samples"]) >= 1
        s0 = data["samples"][0]
        assert len(s0["stroke"]) > 0 and "toe" in s0["true"] and "toe" in s0["pred"]
        assert "toe" in s0["mae"]
        print(f"[self-test] GET /api/test_samples -> {len(data['samples'])} 个样本, "
              f"{len(data['target_names'])} 条曲线/样本  ✅")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[self-test] 失败: {e}")
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("[self-test] 全部通过 ✅" if ok else "[self-test] 存在失败 ❌")
    return ok


def main():
    """命令行入口：解析参数并按需进入自测或常驻服务模式。

    参数：
      --host       监听地址(默认 127.0.0.1)。
      --port       监听端口(默认取环境变量 PORT2，否则 5174)。
      --self-test  仅在线程内启动并自测端点后退出(退出码 0=通过, 1=失败)。
    无 --self-test 时调用 serve() 常驻运行。
    """
    ap = argparse.ArgumentParser(description="悬架运动学代理模型 测试集预测vs真值对照网页")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT2", 5174)))
    ap.add_argument("--self-test", action="store_true", help="线程内启动并自测端点后退出")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
