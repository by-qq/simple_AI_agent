# 时间解析专用 prompt（把相对时间转成 ISO）
class TimePrompts(object):
    TIME_SYSTEM = (
        "你是时间解析器。"
        "请把中文自然语言中的请假时间解析为 ISO 8601 start_time/end_time。"
        "只输出JSON，不要解释。"
    )

    TIME_USER = """现在时间是：{now}
    用户文本：{text}
    
    请输出严格 JSON：
    {{
      "start_time": "YYYY-MM-DD HH:MM 或 null",
      "end_time": "YYYY-MM-DD HH:MM 或 null"
    }}
    
    规则：
    - 能明确推断出具体日期就填 ISO；否则填 null
    - “下周二/明天/后天/本周五”等要结合 now 推断
    - “上午/下午/全天/半天”：
      - 全天：09:00-18:00
      - 上午：09:00-12:00
      - 下午：13:00-18:00
      - 半天：若只说半天且无上下文，按上午 09:00-12:00
    - 如果文本里已经出现 ISO 时间，直接按其输出
    - 不要编造不存在的日期
    - 只输出 JSON
    """
