from .callback import Callback
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)
class TensorboardCallback(Callback):

    # Logger for TensorboardCallback class
    class_logger = logging.getLogger("TensorboardCallback")

    def __init__(self, writer: SummaryWriter,
                 tag: str='loss', 
                 key: str='loss',
                 step_key: str='batch',
                ):
        """   TensorboardX callback to warp visualize 
            scalar after evaluate 

        Args:
            tag: Tag for tensorboardX to log
            key: Key (Metric Name) to log from evaluator
            step_key: log batch or epoch

        """
        self.writer = writer
        self.tag = tag
        self.key = key
        self.step_key = step_key
    
    def __call__(self, engine, *args, **kwargs):
        if self.key not in kwargs:
            TensorboardCallback.class_logger.error(
                f"{self.key} not exists in {kwargs.keys()}. No logging"
            )
            return
        
        if self.step_key not in kwargs:
            TensorboardCallback.class_logger.warning(
                f"{self.step_key} not exists in {kwargs.keys()}"
            )
        
        global_step = kwargs.get(self.step_key, None)
        value = kwargs[self.key]

        self.writer.add_scalar(self.tag, value, global_step)


class LoggerCallback(Callback):
    class_logger = logging.getLogger("LoggerCallback")
    def __init__(self, logger,
                 tag: str='loss', 
                 key: str='loss',
                 step_key: str='batch',
                ):
        """ Logger callback to warp printing 
            scalar after evaluate 

        Args:
            tag: Tag for tensorboardX to log
            key: Key (Metric Name) to log from evaluator
            step_key: log batch or epoch

        """     

        self.logger = logger
        self.tag = tag
        self.key = key
        self.step_key = step_key
        
    def __call__(self, engine, *args, **kwargs):
        if self.key not in kwargs:
            LoggerCallback.class_logger.error(
                f"{self.key} not exists in {kwargs.keys()}. No logging"
            )
            return
        
        if self.step_key not in kwargs:
            LoggerCallback.class_logger.warning(
                f"{self.step_key} not exists in {kwargs.keys()}"
            )

        global_step = kwargs.get(self.step_key, None)
        value = kwargs[self.key]

        self.logger.info(f"{self.step_key}: {global_step}: {self.tag} - {value}")

        