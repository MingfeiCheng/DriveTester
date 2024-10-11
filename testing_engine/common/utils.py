from datetime import datetime
from loguru import logger

def terminal_check(start_time: datetime, run_hour: float) -> bool:
    curr_time = datetime.now()
    t_delta = (curr_time - start_time).total_seconds()
    if t_delta / 3600.0 > run_hour:
        logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
        return True
    return False