from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


from app.deps import get_llm
from app.prompts.HR_prompts import HRPrompts
from app.prompts.TIME_prompts import TimePrompts
from app.models.leave_models import LeaveState
from app.workflows.leave.rules import validate_leave


from app.db.mysql_leave import insert_leave_request, get_leave_balance, get_leave_request, cancel_leave_request, \
    get_recent_leave_requests, update_leave_request, approve_leave_request, reject_leave_request


# 将JSON变成字典
def _safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        return {}

# 检查str表示的时间是否符合ISO格式，转换成datetime并返回
def _safe_iso(s: Any) -> str | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        datetime.fromisoformat(s)
        return s
    except Exception:
        print("日期格式转换失败")
        return None

def _extract_leave_id(text: str) -> str | None:
    if not text:
        return None
    # 正则表达式不要卡的特别si
    m = re.search(r"\bLV-[0-9a-fA-F]{6,12}\b", text)
    return m.group(0) if m else None

def _extract_limit(text:str,default:int = 5):
    if not text:
        return default
    m = re.search(r"(\d+)\s*条", text)
    if not m:
        m = re.search(r"最近\s*(\d+)",text)
    if m :
        try:
            return int(m.group(1))
        except Exception:
            return default
    return default

def decide_intent(state: LeaveState) -> str:
    text = (state.get("text") or state.get("question") or "").lower()

    # cancel intent
    if any(k in text for k in ["取消", "撤销", "作废"]):
        return "cancel"

    # query intent
    if any(k in text for k in ["查询", "查", "状态", "进度", "结果"]):
        if any(k in text for k in ["请假", "年假", "病假", "事假", "休假", "调休", "假期", "申请", "单"]):
            return "query"

    # list intent
    if any(k in text for k in ["最近", "列表", "我的请假", "请假记录", "历史请假"]) and \
            any(k in text for k in ["请假", "年假", "病假", "事假", "休假", "假期", "记录"]):
        return "list"

    # modify intent
    if any(k in text for k in ["修改", "变更", "调整", "改期", "改到", "改为"]):
        return "modify"

    # approve intent (HR/Admin)
    if any(k in text for k in ["批准", "同意", "通过", "审批通过"]):
        return "approve"

    # reject intent (HR/Admin)
    if any(k in text for k in ["驳回", "拒绝", "不通过", "审批拒绝"]):
        return "reject"

    return "apply"

def list_leave_node(state: LeaveState) -> dict:
    text = state.get("text") or state.get("question") or ""
    requester = state.get("requester", "anonymous")
    limit = _extract_limit(text, default=5)

    rows = get_recent_leave_requests(requester, limit=limit)
    if not rows:
        return {"answer": "你还没有请假记录。"}

    lines = [f"最近 {len(rows)} 条请假记录："]
    for r in rows:
        lines.append(
            f"- {r['leave_id']} | {r['leave_type']} | "
            f"{r['start_time']} ~ {r['end_time']} | "
            f"{r['duration_days']}天 | {r['status']}"
        )
    return {"answer": "\n".join(lines)}

def modify_leave_node(state: LeaveState) -> dict:
    text = state.get("text") or state.get("question") or ""
    requester = state.get("requester", "anonymous")

    leave_id = state.get("leave_id") or _extract_leave_id(text)
    if not leave_id:
        return {"answer": "请提供要修改的请假编号（例如 LV-xxxxxxx）。"}

    old = get_leave_request(leave_id)
    if not old:
        return {"answer": f"未找到编号为 {leave_id} 的请假申请。"}
    if old["status"] != "PENDING":
        return {"answer": f"{leave_id} 不是待审批状态，无法修改（当前：{old['status']}）。"}
    # 1) 基于旧单构造 base req
    base_req = {
        "leave_type": old["leave_type"],
        "start_time": old["start_time"].strftime("%Y-%m-%d %H:%M"),
        "end_time": old["end_time"].strftime("%Y-%m-%d %H:%M"),
        "reason": old.get("reason"),
        "requester": old["requester"],
    }

    llm = get_llm()

    # 2) LLM 抽 leave_type / ISO 时间（如果用户给了）
    raw_slots = llm.invoke([
        SystemMessage(content=HRPrompts.SLOT_SYSTEM),
        HumanMessage(content=HRPrompts.SLOT_USER.format(text=text)),
    ]).content
    slots = _safe_json_load(raw_slots)

    # 3) LLM 解析相对时间（如果用户只说“下周二/明天下午”）
    raw_time = llm.invoke([
        SystemMessage(content=TimePrompts.TIME_SYSTEM),
        HumanMessage(content=TimePrompts.TIME_USER.format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M"),
            text=text
        )),
    ]).content
    tdata = _safe_json_load(raw_time)

    new_req = dict(base_req)
    # 优先用 slots 里的 ISO；slots 没有则用相对时间解析结果
    new_req["leave_type"] = slots.get("leave_type") or new_req["leave_type"]

    st = _safe_iso(slots.get("start_time")) or _safe_iso(tdata.get("start_time"))
    et = _safe_iso(slots.get("end_time")) or _safe_iso(tdata.get("end_time"))
    if st:
        new_req["start_time"] = st
    if et:
        new_req["end_time"] = et

    new_req["reason"] = slots.get("reason") or new_req["reason"]

    # 4) validate（余额 + 规则）
    bal = get_leave_balance(requester) or {}
    annual_balance = float(bal.get("annual_days", 0))
    missing, violations = validate_leave(new_req, balance_days=annual_balance)
    if missing or violations:
        tips = []
        if missing:
            tips.append("缺少信息：" + "、".join(missing))
        if violations:
            tips.append("规则问题：" + "；".join(violations))
        return {"answer": "；".join(tips) + "。请重新描述修改内容。"}

    # 5) 落库 update
    ok = update_leave_request(leave_id, {
        "leave_type": new_req["leave_type"],
        "start_time": new_req["start_time"],
        "end_time": new_req["end_time"],
        "duration_days": new_req.get("duration_days"),
        "reason": new_req.get("reason"),
    })
    if not ok:
        return {"answer": "修改失败：该单可能已被审批或取消。"}

    return {
        "leave_id": leave_id,
        "answer": (
            f"已修改请假单 {leave_id}：\n"
            f"- 类型：{new_req['leave_type']}\n"
            f"- 开始：{new_req['start_time']}\n"
            f"- 结束：{new_req['end_time']}\n"
            f"- 原因：{new_req.get('reason') or '无'}"
        )
    }

def intent_node(state: LeaveState) -> dict:
    return {}

# 查询审批节点
def query_leave_node(state: LeaveState) -> dict:
    text = state.get("text") or state.get("question") or ""
    leave_id = state.get("leave_id") or _extract_leave_id(text)

    if not leave_id:
        # 如果没有提供id，就到数据库里查找这个用户所有的或者前面几个请假的单子显示出来
        return {"answer": "请提供请假编号（例如 LV-xxxxxxx），我才能帮你查询。"}

    row = get_leave_request(leave_id)  # 到mysql数据库中按照id查询
    if not row:
        return {"answer": f"未找到编号为 {leave_id} 的请假申请。"}

    return {
        "leave_id": leave_id,
        "answer": (
            f"请假单 {leave_id} 当前状态：{row['status']}\n"
            f"类型：{row['leave_type']}\n"
            f"开始：{row['start_time']}\n"
            f"结束：{row['end_time']}\n"
            f"时长：{row['duration_days']} 天\n"
            f"原因：{row.get('reason') or '无'}"
        ),
    }
# 取消审批节点
def cancel_leave_node(state: LeaveState) -> dict:
    text = state.get("text") or state.get("question") or ""
    leave_id = state.get("leave_id") or _extract_leave_id(text)

    if not leave_id:
        return {"answer": "请提供要取消的请假编号（例如 LV-xxxxxxx）。"}

    ok = cancel_leave_request(leave_id)  # 也是mysql里面写好的取消代码
    if not ok:
        return {"answer": "取消失败：未找到该单，或单据不是待审批状态（PENDING）。"}

    return {"leave_id": leave_id, "answer": f"已取消请假申请 {leave_id}。"}
# 处理时间节点
def parse_time_node(state: LeaveState) -> dict:
    req = state.get("req") or {}
    if _safe_iso(req.get("start_time")) and _safe_iso(req.get("end_time")):
        return {}

    llm = get_llm()
    text = state.get("text", "") or state.get("question", "") or ""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    messages = [
        SystemMessage(content=TimePrompts.TIME_SYSTEM),
        HumanMessage(content=TimePrompts.TIME_USER.format(now=now, text=text)),
    ]
    raw = llm.invoke(messages).content
    # 代码调试的时候，这里一定打断点看看raw返回的是不是你要的结果
    data = _safe_json_load(raw)

    start = _safe_iso(data.get("start_time"))
    end = _safe_iso(data.get("end_time"))

    if start or end:
        req.update({
            "start_time": start or req.get("start_time"),
            "end_time": end or req.get("end_time"),
        })
        return {"req": req}
    return {}

def approve_leave_node(state: LeaveState) -> dict:
    role = (state.get("user_role") or "").lower()
    if role not in {"admin", "hr"}:
        return {"answer": "你没有审批权限（需要 HR/Admin）。"}

    text = state.get("text") or state.get("question") or ""
    leave_id = state.get("leave_id") or _extract_leave_id(text)
    if not leave_id:
        return {"answer": "请提供要审批的请假编号（例如 LV-xxxxxxx）。"}

    ok = approve_leave_request(leave_id, approver=state.get("requester", "admin"))
    if not ok:
        return {"answer": "审批失败：未找到该单，或单据不是待审批状态（PENDING）。"}

    return {"leave_id": leave_id, "answer": f"已审批通过请假单 {leave_id}。"}

def reject_leave_node(state: LeaveState) -> dict:
    role = (state.get("user_role") or "").lower()
    if role not in {"admin", "hr"}:
        return {"answer": "你没有审批权限（需要 HR/Admin）。"}

    text = state.get("text") or state.get("question") or ""
    leave_id = state.get("leave_id") or _extract_leave_id(text)
    if not leave_id:
        return {"answer": "请提供要驳回的请假编号（例如 LV-xxxxxxx）。"}

    # naive reason extraction
    reason = None
    m = re.search(r"(因为|理由|原因)[:： ]?(.*)$", text)
    if m:
        reason = (m.group(2) or "").strip()[:200] or None

    ok = reject_leave_request(
        leave_id,
        approver=state.get("requester", "admin"),
        reason=reason,
    )
    if not ok:
        return {"answer": "驳回失败：未找到该单，或单据不是待审批状态（PENDING）。"}

    return {"leave_id": leave_id, "answer": f"已驳回请假单 {leave_id}。原因：{reason or '未填写'}"}

# 抽槽节点
def extract_slots_node(state: LeaveState) -> dict:
    llm = get_llm()
    text = state.get("text", "") or state.get("question", "") or ""

    messages = [
        SystemMessage(content=HRPrompts.SLOT_SYSTEM),
        HumanMessage(content=HRPrompts.SLOT_USER.format(text=text)),
    ]
    raw = llm.invoke(messages).content
    data = _safe_json_load(raw)

    req = state.get("req") or {}
    req.update({
        "leave_type": data.get("leave_type") or req.get("leave_type"),
        "start_time": _safe_iso(data.get("start_time")) or req.get("start_time"),
        "end_time": _safe_iso(data.get("end_time")) or req.get("end_time"),
        "reason": data.get("reason") or req.get("reason"),
    })
    req["requester"] = state.get("requester", "anonymous")
    return {"req": req}

def validate_node(state: LeaveState) -> dict:
    req = state.get("req") or {}
    requester = req.get("requester") or state.get("requester","anonymous")

    bal = get_leave_balance(requester) or {}    # 看数据库里还有几天假期
    annual_balance = float(bal.get("annual_days",0))

    missing, violations = validate_leave(req, balance_days=annual_balance)
    return {"missing_fields": missing, "violations": violations, "req": req}

def decide_next(state: LeaveState) -> str:
    if state.get("missing_fields") or state.get("violations"):
        return "need_info"
    return "confirm"

def need_info_node(state: LeaveState) -> dict:
    missing = state.get("missing_fields") or []
    violations = state.get("violations") or []

    tips = []
    if missing:
        tips.append("缺少信息：" + "、".join(missing))
    if violations:
        tips.append("规则问题：" + "；".join(violations))

    return {"answer": "；".join(tips) + "。请补充/修正后再说一次。"}

def confirm_node(state: LeaveState) -> dict:
    req = state.get("req") or {}
    ans = (
        "请确认你的请假信息：\n"
        f"- 类型：{req.get('leave_type')}\n"
        f"- 开始：{req.get('start_time')}\n"
        f"- 结束：{req.get('end_time')}\n"
        f"- 时长：{req.get('duration_days')} 天\n"
        f"- 原因：{req.get('reason') or '无'}\n"
        "回复“确认”提交，或直接回复修改后的信息。"
    )
    return {"answer": ans}


def decide_confirm(state: LeaveState) -> str:
    text = (state.get("text") or "").strip().lower()
    if text in {"确认", "确定", "yes", "ok", "submit"}:
        return "create"
    return "end"

def create_leave_node(state: LeaveState) -> dict:
    leave_id = "LV-" + uuid.uuid4().hex[:8]
    req = state.get("req") or dict()
    req_to_save = {
        "leave_id": leave_id,
        "requester": req["requester"],
        "leave_type": req["leave_type"],
        "start_time": req["start_time"],
        "end_time": req["end_time"],
        "duration_days": req["duration_days"],
        "reason": req.get("reason"),
    }
    insert_leave_request(req_to_save)  # 此时这里还没有这个方法，插入mysql数据库
    return {"leave_id": leave_id, "answer": f"已提交请假申请，编号 {leave_id}，等待审批。"}


def build_leave_graph():
    g = StateGraph(LeaveState)

    g.add_node("intent",intent_node)
    g.add_node("query",query_leave_node)
    g.add_node("cancel",cancel_leave_node)
    g.add_node("list",list_leave_node)
    g.add_node("modify", modify_leave_node)
    g.add_node("approve", approve_leave_node)
    g.add_node("reject", reject_leave_node)
    # apply-flow
    g.add_node("parse_time",parse_time_node)
    g.add_node("extract", extract_slots_node)
    g.add_node("validate", validate_node)
    g.add_node("need_info", need_info_node)
    g.add_node("confirm", confirm_node)
    g.add_node("create", create_leave_node)


    g.add_edge(START, "intent")
    g.add_conditional_edges(
        "intent",
        decide_intent,
        {
            "apply": "parse_time",
            "query": "query",
            "cancel": "cancel",
            "list": "list",
            "modify": "modify",
            "approve": "approve",
            "reject": "reject",
        },
    )
    g.add_edge("parse_time", "extract")
    g.add_edge("extract", "validate")

    g.add_conditional_edges(
        "validate",
        decide_next,
        {
            "need_info": "need_info",
            "confirm": "confirm"
        },
    )

    g.add_conditional_edges(
        "confirm",
        decide_confirm,
        {
            "create": "create",
            "end": END
        },
    )

    g.add_edge("query", END)
    g.add_edge("cancel", END)
    g.add_edge("list", END)
    g.add_edge("modify", END)
    g.add_edge("approve", END)
    g.add_edge("reject", END)
    g.add_edge("need_info", END)
    g.add_edge("create", END)

    return g.compile()
