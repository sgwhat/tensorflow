import argparse
import time
import re
import numpy as np
import pandas as pd
from io import StringIO
import os
import subprocess
from contextlib import contextmanager
from argparse import RawTextHelpFormatter

DNNL_VERBOSE = False if 0 == os.environ.get("DNNL_VERBOSE", 0) else True

COLS = [
    "marker", "type", "engine", "primitive", "impl", "prop", "iotypes",
    "postops", "aux", "prob", "elapsed"
]
UNIQ = [x for x in COLS if x not in ('marker', 'type', 'elapsed')]


class OneDNNLog:
    r"""Creates DNN log object from DNN verbose log file"""

    def __init__(self, fname, is_iters=True, start=0, end=-1):
        self.fname = fname
        self.is_iters = is_iters
        self.start = start
        self.end = end
        self.df = pd.DataFrame()
        self.df = self.get_df()

    def get_df(self):
        r"""
        Convert a DNN verbose log file into pandas dataframe, 
        adds unique kernel identifiers (uid), and iterations.
        """
        if not self.df.empty:
            return self.df
        iters = []
        lines = []
        self.idf = pd.DataFrame(
            columns=['event', 'dur', 'ts', 'pid', 'tid', 'args'])
        with open(self.fname) as f:
            # lines = [l for l in f if "verbose,exec" in l]
            i = 0
            for l in f:
                if "parsednn_event,begin,iteration_" in l:
                    prev = len(lines)
                    il = l.split(",")
                if "parsednn_event,end,iteration_" in l:
                    iters.append((prev, len(lines)))
                    il2 = l.strip().split(",")
                    self.idf.loc[i] = (il2[2], float(il2[3]) - float(il[3]),
                                       il[3], il2[4], il2[5],
                                       ",".join(il2[6:]))
                    i += 1
                if "verbose,exec" in l:
                    lines.append(l)
        if len(lines) == 0:
            return self.df
        fields = lines[0].split(",")
        global COLS
        global UNIQ
        if "mkldnn" in fields[0]:
            COLS = [x for x in COLS if x not in ('engine', 'aux')]
            UNIQ = [x for x in UNIQ if x not in ('engine', 'aux')]
        if len(fields) == len(COLS) + 1:
            COLS.append("start")
        df = pd.read_csv(StringIO("\n".join(lines)), names=COLS)
        # df = df[(df["elapsed"].notna()) & (df["type"] == "exec")]
        if df.empty:
            return df

        def uid(v):
            if isinstance(v[5], str):
                v[5] = re.sub(r'\d+', '', v[5])
            hv = hash(tuple(v))
            r = uid.h.get(hv, -1)
            if r == -1:
                uid.h[hv] = uid.c
                uid.c += 1
                return uid.c - 1
            else:
                return r

        uid.c, uid.h = 0, {}
        df["uid"] = df.apply(lambda k: uid(list(k.loc[UNIQ])), axis=1)
        self.df = df.reset_index(drop=True)
        self.num_iters = 1
        if self.start != 0 or self.end != -1:
            self.is_iters = False
            self.df = self.df.iloc[start:end]
        self.df["iter"] = 0
        if self.is_iters:
            iters = iters if len(iters) > 0 else self._get_iter_tuples()
            for i, x in enumerate(iters):
                self.df.loc[x[0]:x[1], "iter"] = i + 1
            self.num_iters = len(iters) + 1
        return self.df

    def _get_iter_tuples(self):
        # try to find iterations
        ops = self.df.loc[self.df["primitive"] != "reorder"]
        firstop = self.query(ops.index[0])
        lastop = self.query(ops.index[-1])
        if len(firstop) != len(lastop) or len(lastop) == 1:
            return [(0, len(self.df) - 1)]
        self.num_iters = len(lastop)

        ii, ij = lastop.iloc[-2].name, lastop.iloc[-1].name
        for i in range(ii + 1, ij):
            q = self.query(i)
            if len(q) != self.num_iters:
                continue
            if q.iloc[0].name > lastop.iloc[0].name:
                continue
            r = [(0, q.index[1] - 1)]
            for j, x in enumerate(q.index[1:-1]):
                r.append((x, q.index[j + 2] - 1))
            r.append((q.index[-1], self.df.index[-1]))
            return r
        return [(0, len(self.df) - 1)]

    def update_iters(self, itrace):
        r"""Use iterations trace to idenitfy and filter iterations"""
        itrace = pd.read_json(itrace).sort_values('ts', ignore_index=True)
        stime = itrace.loc[0, "ts"] / 1e3
        self.df = self.df[self.df["start"] > stime]
        for i, j in itrace.iterrows():
            stime = itrace.loc[i, "ts"] / 1e3
            etime = stime + (itrace.loc[i, "dur"] / 1e3)
            self.df.loc[(self.df["start"] > stime) &
                        (self.df["start"] < etime), "iter"] = i
        self.num_iters = len(itrace)

    def query(self, i):
        r"""Query DNN log for all occurances that match row i"""
        k = self.df.iloc[i]["uid"]
        return self.df.query('uid==@k')

    def get_uid(self, uid):
        return self.df[self.df["uid"] == uid].iloc[0]

    def compare(self, i, j):
        r"""Query if row i and j have same DNN kernel execution"""
        return self.df.loc[i, UNIQ].equals(self.df.loc[j, UNIQ])

    def iter(self, iter=0):
        r"""return dataframe for iteration i"""
        if not self.is_iters:
            return self.df
        return self.df.loc[self.df["iter"] == (
            iter if iter >= 0 else self.num_iters + iter)]

    def summary(self, iter=0):
        r"""Summary of the DNN log for given iteration"""
        return self.iter(iter).groupby(['primitive']).agg(
            total_ms=("elapsed", "sum"),
            count=("primitive", "count")).sort_values(by="total_ms",
                                                      ascending=False)

    def detail(self, iter=0, n=20):
        r"""Detailed view of the DNN log for given iteration"""
        iter = self.iter(iter)
        df = iter.groupby(['uid']).agg(
            count=("primitive", "count"),
            avg_ms=("elapsed", "mean"),
            total_ms=("elapsed", "sum")).sort_values(by="total_ms",
                                                     ascending=False).head(n)
        b = Bench(iter)
        df.insert(
            0, 'shape_in',
            df.apply(lambda x: b.get_short_shapes(self.get_uid(x.name)),
                     axis=1))
        df.insert(
            0, 'kernel',
            df.apply(lambda x: b.get_short_name(self.get_uid(x.name)), axis=1))
        return df

    def iters(self):
        r"""Summarize all the iterations found in DNN log"""
        return self.df.pivot_table(index=["iter"],
                                   columns="primitive",
                                   values="elapsed",
                                   aggfunc=['sum', 'count'])


class Bench:
    r"""Parse DNN log to create Benchdnn mapping and execution"""
    DIR = {
        "forward_training": "FWD_D",
        "forward_training_bias": "FWD_B",
        "forward_inference": "FWD_D",
        "forward_inference_bias": "FWD_B",
        "forward_training_bias": "FWD_B",
        "backward_data": "BWD_D",
        "backward_weights": "BWD_W",
        "backward_weights_bias": "BWD_WB",
        "backward": "BWD_DW"
    }
    P2P = {
        "convolution": "conv",
        "deconvolution": "deconv",
        "inner_product": "ip",
        "batch_normalization": "bnorm",
        "pooling": "pool",
        "matmul": "matmul",
        "eltwise": "eltwise",
        "reorder": "reorder",
        "sum": "sum",
        "softmax": "softmax",
        "binary": "binary",
    }

    def __init__(self, df):
        if "uid" not in df.columns:
            raise TypeError(f'invalid OneDNN dataframe')
        self.df = df
        self.kernels = pd.DataFrame(columns=['primitive', 'kernel', 'count'])

    def get_kernels(self):
        if not self.kernels.empty:
            return self.kernels
        for i, j in self.df.groupby(["uid"
                                     ]).agg("count")["primitive"].iteritems():
            s = self.df[self.df["uid"] == i].iloc[0]
            if i in self.kernels.index:
                continue
            self.kernels.loc[i] = (s["primitive"],
                                   getattr(Bench, s["primitive"],
                                           Bench.default)(self, s), j)
        return self.kernels

    def get_prefix(self, s):
        return f'--{Bench.P2P[s["primitive"]]} --mode=p{self.get_engine(s)}'

    def get_types(self, s):
        r = {}
        t = s["iotypes"].split(" ")
        for x in t:
            dt = x.split(":")
            i, v = dt[0].split("_") if "_" in dt[0] else (dt[0], "")
            r[i] = (v, dt[1:]) if len(dt) > 1 else (v, [])
        return r

    def get_alg(self, s):
        alg = ""
        if "aux" not in s:
            return alg
        for x in s["aux"].split(" "):
            i, j = x.split(":")
            j = '_'.join(j.split("_")[1:]) if "_" in j else j
            alg = alg + " --" + i + "=" + j
        return alg

    def get_cfg(self, s):
        t = self.get_types(s)
        if "f32" == t.get("src", [""])[0] == t.get("dst", [""])[0]:
            return " --cfg=f32"
        if "f16" == t.get("src", [""])[0] == t.get("dst", [""])[0]:
            return " --cfg=f16"
        if ("u8" == t.get("src", [""])[0]
                or "s8" == t.get("src", [""])[0]) and "wei" in t:
            return f' --cfg={t.get("src", [""])[0]}{t.get("wei", [""])[0]}{t.get("dst", [""])[0]}'
        if "bf16" == t.get("src", [""])[0]:
            return f' --cfg={t.get("src", [""])[0]}{t.get("wei", [""])[0]}{t.get("dst", [""])[0]}'
        if s["primitive"] == "pooling":
            if t.get("src", [""])[0] == t.get("dst", [""])[0]:
                return f' --cfg={t.get("src", [""])[0]}'
            else:
                return f' --cfg={t.get("src", [""])[0]}{t.get("dst", [""])[0]}'
        if "fsrc" in t:
            return ""
        print(s)
        raise TypeError(f'{t}\nUnsupported cfg')

    def get_dt_prefix(self, s):
        if s["primitive"] in ('convolution', 'deconvolution', 'inner_product',
                              'matmul', 'pooling'):
            return self.get_cfg(s)
        if s["primitive"] in ("batch_normalization", "eltwise", "softmax"):
            return self.get_dt(s)
        if s["primitive"] in ("reorder", "sum"):
            t = self.get_types(s)
            return f' --sdt={t["src"][0]} --ddt={t["dst"][0]}'
        if s["primitive"] in ("binary"):
            t = self.get_types(s)
            return f' --sdt={t["src"][0]}:{t["src"][0]} --ddt={t["dst"][0]}'

    def get_dir(self, s):
        t = self.get_types(s)
        dir = s["prop"]
        if dir == "undef":
            return dir
        bia = t.get("bia", [""])[0]
        if bia != "" and bia != "undef":
            dir = dir + "_bias"
        if dir not in Bench.DIR:
            print(s)
            raise TypeError(f'{t} {dir}\nUnsupported prop found')
        return Bench.DIR[dir]

    def get_postops_str(self, s):
        ps = ""
        if isinstance(s["postops"], str) and "post_ops" in s["postops"]:
            ps = s["postops"].split("post_ops:")[1].replace("eltwise_",
                                                            "")[:-1]
        return ps

    def get_postops(self, s):
        p = ""
        if self.get_postops_str(s) != "":
            p = f' --attr-post-ops="{ps}"'
        return p

    def get_engine(self, s):
        e = ""
        if "engine" in s:
            e = f' --engine={s["engine"]}'
        return e

    def get_short_shapes(self, s):
        prob = s["prob"].split("_")
        sprob = [
            re.search("\d+", x).group() for x in prob if not x[0].isdigit()
        ]
        prob = sprob if len(sprob) > 0 else prob
        return "x".join(prob)

    def get_short_name(self, s):
        names = []
        names.append(Bench.P2P.get(s["primitive"], s["primitive"]))
        dir = self.get_dir(s)
        if dir != "undef":
            names.append(dir.lower())
        ps = self.get_postops_str(s)
        if ps != "":
            ps = ps.replace("'", "").replace(";", "_")
            if ps[-1] == "_":
                ps = ps[:-1]
            names.append(ps)
        return "_".join(names)

    def convimpl(self, s):
        return f'--dir={self.get_dir(s)}{self.get_postops(s)} {s["prob"]}'

    def convolution(self, s):
        return self.convimpl(s)

    def deconvolution(self, s):
        return self.convimpl(s)

    def inner_product(self, s):
        return self.convimpl(s)

    def matmul(self, s):
        t = self.get_types(s)
        bia = ""
        if "bia" in t:
            bia = bia + " --bia_dt=" + t["bia"][0]
            m = [x for x in t["bia"][1] if "mask" in x]
            mask = m[0].split("mask") if m else ""
            bia = bia + " --bia_mask=" + mask[1]
        return f'{bia}{self.get_postops(s)} {s["prob"]}'

    def pooling(self, s):
        alg = self.get_alg(s)
        return f'--dir={self.get_dir(s)}{alg}{self.get_postops(s)} {s["prob"]}'

    def get_dt(self, s):
        t = self.get_types(s)
        dt = f' --dt={t["data"][0]}' if "data" in t else f''
        return dt

    def batch_normalization(self, s):
        fv = s["aux"].split(":")
        f = "" if len(fv) < 2 else fv[1]
        return f'--flags={f} --dir={self.get_dir(s)}{self.get_postops(s)} {s["prob"]}'

    def eltwise(self, s):
        alg = self.get_alg(s)
        return f'--dir={self.get_dir(s)}{alg} {s["prob"]}'

    def binary(self, s):
        alg = self.get_alg(s)
        return f'{alg}{self.get_postops(s)} {s["prob"].split(" ")[0]}'

    def get_sdtag(self, s):
        t = self.get_types(s)
        if "in" in t:
            return f''
        st = t["src"][1][2]
        dt = t["dst"][1][2]
        return f'--stag={st} --dtag={dt}'

    def reorder(self, s):
        t = self.get_types(s)
        st = t["src"][1][2]
        dt = t["dst"][1][2]
        return f'{self.get_sdtag(s)} {s["prob"]}'

    def sum(self, s):
        t = self.get_types(s)
        return f'{self.get_sdtag(s)} {s["prob"]}'

    def softmax(self, s):
        axis = ""
        if "aux" in s:
            axis = f' --axis={s["aux"].split("axis:")[1]}'
        return f'--dir={self.get_dir(s)}{axis} {s["prob"]}'

    def default(self, s):
        return "notsupported"

    def summary(self):
        if len(self.get_kernels()) == 0:
            raise TypeError(f'execution list is empty')

        def _run(ik):
            ik["benchdnn command"] = self.bench_cmd(ik)
            return ik

        self.kernels = self.get_kernels().apply(_run, axis=1)
        return self.kernels

    def bench_cmd(self, ik):
        s = self.df.loc[self.df["uid"] == ik.name].iloc[0]
        return self.get_prefix(s) + self.get_dt_prefix(s) + " " + ik["kernel"]

    def batch(self, p=None):
        if not p:
            return self.get_kernels()
        return self.kernels[self.get_kernels()["primitive"] == p]

    def run(self):
        if len(self.get_kernels()) == 0:
            raise TypeError(f'execution list is empty')

        def _run(ik):
            cmd = self.bench_cmd(ik)
            print(cmd)
            r = subprocess.run(["benchdnn"] + cmd.split(" "),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
            l = r.stdout.split("\n")
            t, v = 0, 0
            if len(l) > 1 and "perf" == l[1][:4]:
                t, v = l[1].split(",")[-2:]
            ik["avg_ms"], ik["gflops"] = t, v
            return ik

        self.kernels = self.get_kernels().apply(_run, axis=1)
        return self.kernels


@contextmanager
def LogEvent(enabled=True,
             event="iteration_n",
             tid="Python",
             pid="Python",
             args=""):
    r"""
    Prints given event with begin and end timestamps
    OneDNNLog parser can use "iteration_" prefix as iteration markers.
    """
    args = str(args).replace('\n', '')
    def p(t): return print(
        f'parsednn_event,{t},{event},{time.time()*1e6},{pid},{tid},{args}', flush=True)
    if DNNL_VERBOSE and enabled:
        p("begin")
    try:
        yield
    finally:
        if DNNL_VERBOSE and enabled:
            p("end")


def process():
    parser = argparse.ArgumentParser(description='OneDNN verbose log stats',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f',
                        '--file',
                        default='./dnn.log',
                        help='file with dnnl_verbose log (default: dnn.log)')
    parser.add_argument(
        '-r',
        type=int,
        default=-1,
        help='which iteration to use for summarizing (default: last iter)')
    parser.add_argument('-d',
                        '--detail',
                        action='store_true',
                        default=False,
                        help='show details of top kernels')
    parser.add_argument(
        '-n',
        type=int,
        default=20,
        help='show n number of top kernels in detail (default 20, use -1 for all)')
    parser.add_argument('-i',
                        '--iters',
                        action='store_true',
                        default=False,
                        help='show iterations')
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help=f'use index in dump as start, not iteration (default: 0)')
    parser.add_argument(
        '--end',
        type=int,
        default=-1,
        help=f'use index in dump as end, not iteration (default: -1)')
    parser.add_argument(
        '-b',
        '--bench',
        nargs=2,
        help=f'benchdnn show cmd, batch, or run. eg:\n "-b cmd all"\n "-b batch matmul"\n "-b run matmul"'
    )
    parser.add_argument('-t',
                        '--trace',
                        action='store_true',
                        default=False,
                        help='create chrome trace, save to ptrace.json')
    parser.add_argument('--ftrace', help='append framework trace')
    parser.add_argument('--noiters',
                        action='store_true',
                        default=False,
                        help='use entire log as 1 iteration')
    parser.add_argument('--dump',
                        action='store_true',
                        default=False,
                        help='dump OneDNN dataframe')

    args = parser.parse_args()
    d = OneDNNLog(args.file, not args.noiters, args.start, args.end)
    if d.get_df().empty:
        print(f'Invalid OneDNN log file')
        return

    if args.dump:
        print(d.get_df().loc[:, COLS].to_string())
        return

    if args.bench:
        c, t = args.bench
        bd = d.iter(args.r)
        if t != "all":
            bd = bd[bd["primitive"] == t]
        b = Bench(bd)
        if c == "cmd":
            b = b.summary().loc[:, ["benchdnn command", "count"]]
            print(b.to_string())
        if c == "batch":
            b = b.batch()
            for i, x in b.iterrows():
                print(f'{x["kernel"]}n"*{x["count"]}"')
        if c == "run":
            b = b.run()
            print(b.to_string())
        return

    if args.detail:
        pd.set_option('display.max_colwidth', None)
        s = d.detail(args.r, args.n).round(decimals=2)
        print(s.to_string())
        print(f'\ntotal time in OneDNN {s.sum()["total_ms"]:.2f}ms')
        return
    if args.iters:
        s = d.iters().round(decimals=4)
        s.loc[:, 'ops'] = s.loc[:, "count"].sum(axis=1).astype(int)
        s.loc[:, 'total_ms'] = s.loc[:, "sum"].sum(axis=1)
        s = s.rename(columns={"sum": "time_ms"})
        print(s.loc[:, ~s.columns.get_level_values(0).isin({"count"})])
        print(f'\ntotal time in OneDNN {s.sum()["total_ms",""]:.2f}ms')
        return

    s = d.summary(args.r).round(decimals=4)
    print(s.to_string())
    print(f'\ntotal time in OneDNN {s.sum()["total_ms"]:.2f}ms')

    if args.trace:
        if "start" not in d.df.columns:
            print(f'\nOneDNN Log does not have timestamp')
            return
        irecs = "[]"
        frecs = "[]"
        stime = 0
        if not d.idf.empty:
            stime = float(d.idf.loc[0, "ts"])
            d.df = d.df[d.df["start"] * 1e3 > stime]
            lirecs = []
            for i, j in d.idf.iterrows():
                lirecs.append(
                    f'{{"name": "{j["event"]}", "ph": "X", "ts": {j["ts"]}, "dur": {j["dur"]}, "tid": "{j["tid"]}", "pid": "{j["pid"]}", "args": {{}}}}'
                )
            irecs = "[" + ",".join(lirecs) + "]"

        if args.ftrace:
            ftrace = pd.read_json("trace.json")
            # check if TF trace
            if "traceEvents" in ftrace:
                ftrace = ftrace["traceEvents"]
            else:
                if stime == 0:
                    raise TypeError("Cannot adjust framework trace time")
                ftrace["ts"] = ftrace.ts.apply(lambda x: x + stime)
            frecs = ftrace.to_json(orient='records')
        recs = []
        for i, j in d.df.iterrows():
            recs.append(
                f'{{"name": "{j["primitive"]}", "ph": "X", "ts": {j["start"]*1e3}, "dur": {j["elapsed"]*1e3}, "tid": "OneDNN", "pid": "OneDNN", "args": {{}}}}'
            )
        with open("ptrace.json", "w") as f:
            il, fl = len(irecs), len(frecs)

            if il > 2 and fl > 2:
                f.write(irecs[:-1] + "," + frecs[1:-1] + "," + ",".join(recs) +
                        ']')
            elif il > 2:
                f.write(irecs[:-1] + "," + ",".join(recs) + ']')
            elif fl > 2:
                f.write(frecs[:-1] + "," + ",".join(recs) + ']')
            else:
                f.write('[' + ",".join(recs) + ']')
            print(f'\nTrace written to ptrace.json')


if __name__ == '__main__':
    process()
