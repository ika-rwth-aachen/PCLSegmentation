# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from nets.Darknet import Darknet
from nets.SqueezeSegV2 import SqueezeSegV2
from configs.SqueezeSegV2 import SqueezeSegV2Config
from configs.Darknet52 import Darknet52
from configs.Darknet21 import Darknet21

model_map = {
  "squeezesegv2": SqueezeSegV2,
  "darknet52": Darknet,
  "darknet21": Darknet
}

config_map = {
  "squeezesegv2": SqueezeSegV2Config,
  "darknet52": Darknet52,
  "darknet21": Darknet21
}


def load_model_config(model_name):
  config = config_map[model_name.lower()]()
  model = model_map[model_name.lower()](config)
  return config, model
