#!/usr/bin/env python3
"""Offline numerical sanity checks for calibration fixtures and profiles."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/calibration/holdout-overconfident.json"
PROFILE = ROOT / "tests/calibration/profile-overconfident.json"


def softmax(logits, t):
    xs=[x/t for x in logits]; m=max(xs); ws=[math.exp(x-m) for x in xs]; z=sum(ws)
    return [w/z for w in ws]


def metrics(examples, t, bins_n=15):
    total=correct=loss=brier=0.0
    bins=[dict(count=0,weight=0.0,conf=0.0,correct=0.0) for _ in range(bins_n)]
    for ex in examples:
        w=float(ex.get("weight",1.0)); ps=softmax(ex["logits"],t); target=ex["target"]
        pred=max(range(len(ps)),key=lambda i: ps[i]); confidence=ps[pred]; ok=1.0 if pred==target else 0.0
        total += w; correct += w*ok; loss += w * -math.log(max(ps[target], float.fromhex('0x1.0p-1022')))
        brier += w * sum((p-(1.0 if i==target else 0.0))**2 for i,p in enumerate(ps))
        bi=min(int(math.floor(confidence*bins_n)),bins_n-1); b=bins[bi]
        b["count"]+=1; b["weight"]+=w; b["conf"]+=w*confidence; b["correct"]+=w*ok
    ece=0.0; rows=[]
    for i,b in enumerate(bins):
        if b["weight"]:
            mean_confidence=b["conf"]/b["weight"]; accuracy=b["correct"]/b["weight"]; gap=abs(mean_confidence-accuracy); ece += b["weight"]/total*gap
        else: mean_confidence=accuracy=gap=0.0
        rows.append(dict(lower=i/bins_n,upper=(i+1)/bins_n,count=b["count"],weight=b["weight"],mean_confidence=mean_confidence,accuracy=accuracy,gap=gap))
    return dict(count=len(examples),total_weight=total,accuracy=correct/total,nll=loss/total,brier=brier/total,ece=ece,ece_bins=rows)


def fit(examples, lo=.05, hi=20.0, iters=96):
    left,right=math.log(lo),math.log(hi); ratio=(math.sqrt(5)-1)/2
    c=right-ratio*(right-left); d=left+ratio*(right-left)
    fc=metrics(examples,math.exp(c),1)["nll"]; fd=metrics(examples,math.exp(d),1)["nll"]
    for _ in range(iters):
        if fc <= fd:
            right,d,fd=d,c,fc; c=right-ratio*(right-left); fc=metrics(examples,math.exp(c),1)["nll"]
        else:
            left,c,fc=c,d,fd; d=left+ratio*(right-left); fd=metrics(examples,math.exp(d),1)["nll"]
    return math.exp((left+right)/2)


def assert_close(a,b,tol=1e-10):
    assert abs(float(a)-float(b)) <= tol * max(1.0,abs(float(a)),abs(float(b))), (a,b)


def main():
    data=json.loads(DATA.read_text()); profile=json.loads(PROFILE.read_text())
    assert data["contract"] == "turbojev.calibration-dataset.v1"
    assert profile["contract"] == "turbojev.calibration-profile.v1"
    assert profile["method"] == "temperature-softmax"
    assert len(data["examples"]) == profile["sample_count"] == 10
    for ex in data["examples"]:
        assert len(ex["logits"]) >= 2
        assert 0 <= ex["target"] < len(ex["logits"])
        assert all(math.isfinite(float(x)) for x in ex["logits"])
        assert float(ex.get("weight",1.0)) > 0
    t=fit(data["examples"]); before=metrics(data["examples"],1.0,15); after=metrics(data["examples"],t,15)
    assert after["nll"] < before["nll"]
    assert_close(profile["temperature"],t)
    for name, expected in [("before",before),("after",after)]:
        got=profile[name]
        for key in ("accuracy","nll","brier","ece","total_weight"):
            assert_close(got[key],expected[key])
        assert got["count"] == expected["count"]
        assert len(got["ece_bins"]) == 15
    print(f"OK: calibration profile temperature={t:.9f}, NLL {before['nll']:.6f} -> {after['nll']:.6f}, ECE {before['ece']:.6f} -> {after['ece']:.6f}")

if __name__ == "__main__": main()
