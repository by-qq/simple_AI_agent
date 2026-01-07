from __future__ import annotations

from celery import Celery
from app.config import settings

celery_app = Celery("enterprise_kb_assistant", broker=settings.celery_broker_url)

celery_app.autodiscover_tasks(["app.tasks"])
# 创建了一个celery_app后，让它自动发现任务，我们有一个文件夹app/tasks专门存放各种任务

# 如果没自动发现，我们只好手动注册一个任务到celery_app中,防没找到
_existing = tuple(celery_app.conf.get("imports") or ())  		# 主要加这里的3行
if "app.tasks.audio_tasks" not in _existing:
    celery_app.conf.imports = _existing + ("app.tasks.audio_tasks",)

# 对celery_app中的一些默认配置做调整
celery_app.conf.update(
    task_acks_late=True,                # 任务完成后才给我一个应答ack就是应答
    task_reject_on_worker_lost=True,    # 重新投递任务，在我们这里就是文件上传失败不要卡着，而是重新投递任务
    worker_prefetch_multiplier=1,       # 每个worker领1个任务，多半和task_acks_late搭配使用
    timezone="UTC",
)

