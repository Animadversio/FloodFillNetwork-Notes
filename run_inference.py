# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.
"""

import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
import logging
import logging.config
logger = logging.getLogger(__name__)
# logging.config.fileConfig(json.load(open('configs/logging.json')), disable_existing_loggers=False)
FLAGS = flags.FLAGS
flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')
flags.DEFINE_string('logfile_name', "inference_log1.log", "Log file to write the log instantaneously")
FLAGS = flags.FLAGS


def main(unused_argv):
  logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(filename)s: %(funcName)s(): %(lineno)d] %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "logfile": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": FLAGS.logfile_name,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "logfile"],
                "level": "INFO",
                "propagate": True
            }
        }
    })
  logging.info("Logger prepared! ")
  request = inference_flags.request_from_flags()  # Parsed structure from flag

  if not gfile.Exists(request.segmentation_output_dir):
    gfile.MakeDirs(request.segmentation_output_dir)

  bbox = bounding_box_pb2.BoundingBox()  # bounding box structure
  text_format.Parse(FLAGS.bounding_box, bbox)  # Parse the param in flag

  runner = inference.Runner()
  runner.start(request)
  runner.run((bbox.start.z, bbox.start.y, bbox.start.x),
             (bbox.size.z, bbox.size.y, bbox.size.x))  # Main Body

  counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
  if not gfile.Exists(counter_path):
    runner.counters.dump(counter_path)
  runner.stop_executor()

if __name__ == '__main__':
  app.run(main)

