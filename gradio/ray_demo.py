from re import T
import gradio as gr
import ray
from ray import serve
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple
import asyncio

# integration API
def start_ray_backend():
    ray.init(num_cpus=10)
    return serve.start(detached=True)

def rayify_interface(serve_client, io: gr.Interface, names=None):
    # note: the rayify process makes serve deployments, before gradio launches
    if names is None:
        names = [fn.__name__ for fn in io.predict]
    assert len(io.predict) == len(names)
    
    wrapped_fns = []
    for (predict_fn, name) in zip(io.predict, names):
        model = serve.deployment(name=name, num_replicas=2)(predict_fn)
        model.deploy()
        handle = serve_client.get_handle(name)

        wrapped_fns.append(lambda *args: asyncio.wrap_future(handle.remote(*args).future()))
    io.predict = wrapped_fns
    return io

# Gradio functions
def greet(name):
    return "Hello " + name + "!!"

def goodbye(name):
    return "See you later " + name + "!!"

# how it works!
serve_client = start_ray_backend()
demo = gr.Parallel(
    rayify_interface(serve_client, gr.Interface(greet, "text", "text")),
    rayify_interface(serve_client, gr.Interface(goodbye, "text", "text"))
)
demo.launch()