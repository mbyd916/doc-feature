#coding=utf-8
import logging
from logging.config import fileConfig

logging.config.fileConfig("log.conf")
logger = logging.getLogger()