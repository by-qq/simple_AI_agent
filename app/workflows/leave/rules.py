from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List


def validate_leave(req: Dict[str,Any], balance_days:float=5.0) ->Tuple[List[str],List[str]]:
    missing = []
    violations = [] # 错误信息

    for f in ["leave_type","start_time","end_time"]:
        if not req.get(f):  # 空信息就添加到缺失列表中
            missing.append(f)

    if missing:
        return missing, violations

    # parse time
    try:# 对时间进行解析必须抛异常
        # 所有从前台拿到的的内容全部都是str类型，即使是数字也是str
        start = datetime.fromisoformat(req["start_time"])
        end = datetime.fromisoformat(req["end_time"])
    except Exception as exc:
        violations.append("start_time/end_time格式应该为ISO（YYYY-MM-DD HH:MM）")
        return missing, violations

    if end <= start:
        # 这里这些提示字符串需要写一个类单独存放
        violations.append("结束时间必须晚于开始时间")

    duration = (end - start).total_seconds() / 3600.0 /8.0
    if duration < 0.5:
        violations.append("最小请假单位是0.5天")

    leave_type = req.get("leave_type")
    if leave_type =="annual":
        if duration > balance_days:
            violations.append("余额不足")
        if start < datetime.now() + timedelta(days=1):
            violations.append("年假需要至少提前一个工作日提交")

    if leave_type =="sick":
        if duration > 1 and not req.get("reason"):
            violations.append("病假超过一天需要提供证明")

    req["duration_days"]=round(duration,2)

    return missing,violations







