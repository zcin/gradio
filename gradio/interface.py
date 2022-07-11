"""
This is the core file in the `gradio` package, and defines the Interface class,
including various methods for constructing an interface and then launching it.
"""

from __future__ import annotations

import copy
import csv
import inspect
import json
import os
import random
import re
import warnings
import weakref
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin

from gradio import interpretation, utils
from gradio.blocks import Blocks
from gradio.components import (
    Button,
    Component,
    Dataset,
    Interpretation,
    IOComponent,
    Markdown,
    StatusTracker,
    Variable,
    get_component_instance,
)
from gradio.events import Changeable, Streamable
from gradio.external import load_from_pipeline  # type: ignore
from gradio.flagging import CSVLogger, FlaggingCallback  # type: ignore
from gradio.layouts import Column, Row, TabItem, Tabs
from gradio.process_examples import cache_interface_examples, load_from_cache

import ray
from ray import serve
import asyncio

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    import transformers


class Interface(Blocks):
    """
    The Interface class is a high-level abstraction that allows you to create a
    web-based demo around a machine learning model or arbitrary Python function
    by specifying: (1) the function (2) the desired input components and (3) desired output components.
    """

    # stores references to all currently existing Interface instances
    instances: weakref.WeakSet = weakref.WeakSet()

    class InterfaceTypes(Enum):
        STANDARD = auto()
        INPUT_ONLY = auto()
        OUTPUT_ONLY = auto()
        UNIFIED = auto()

    @classmethod
    def get_instances(cls) -> List[Interface]:
        """
        :return: list of all current instances.
        """
        return list(Interface.instances)

    @classmethod
    def load(
        cls,
        name: str,
        src: Optional[str] = None,
        api_key: Optional[str] = None,
        alias: Optional[str] = None,
        **kwargs,
    ) -> Interface:
        """
        Class method that constructs an Interface from a Hugging Face repo. Can accept
        model repos (if src is "models") or Space repos (if src is "spaces"). The input
        and output components are automatically loaded from the repo.
        Parameters:
        name (str): the name of the model (e.g. "gpt2"), can include the `src` as prefix (e.g. "models/gpt2")
        src (str | None): the source of the model: `models` or `spaces` (or empty if source is provided as a prefix in `name`)
        api_key (str | None): optional api key for use with Hugging Face Hub
        alias (str | None): optional string used as the name of the loaded model instead of the default name
        Returns:
        (gradio.Interface): a Gradio Interface object for the given model
        """
        return super().load(name=name, src=src, api_key=api_key, alias=alias, **kwargs)

    @classmethod
    def from_pipeline(cls, pipeline: transformers.Pipeline, **kwargs) -> Interface:
        """
        Class method that constructs an Interface from a Hugging Face transformers.Pipeline object.
        The input and output components are automatically determined from the pipeline.
        Parameters:
        pipeline (transformers.Pipeline): the pipeline object to use.
        Returns:
        (gradio.Interface): a Gradio Interface object from the given Pipeline
        """
        interface_info = load_from_pipeline(pipeline)
        kwargs = dict(interface_info, **kwargs)
        interface = cls(**kwargs)
        return interface

    def __init__(
        self,
        fn: Callable | List[Callable],
        inputs: Optional[str | Component | List[str | Component]],
        outputs: Optional[str | Component | List[str | Component]],
        examples: Optional[List[Any] | List[List[Any]] | str] = None,
        cache_examples: Optional[bool] = None,
        examples_per_page: int = 10,
        live: bool = False,
        interpretation: Optional[Callable | str] = None,
        num_shap: float = 2.0,
        title: Optional[str] = None,
        description: Optional[str] = None,
        article: Optional[str] = None,
        thumbnail: Optional[str] = None,
        theme: Optional[str] = None,
        css: Optional[str] = None,
        allow_flagging: Optional[str] = None,
        flagging_options: List[str] = None,
        flagging_dir: str = "flagged",
        flagging_callback: FlaggingCallback = CSVLogger(),
        analytics_enabled: Optional[bool] = None,
        _repeat_outputs_per_model: bool = True,
        **kwargs,
    ):
        """
        Parameters:
        fn (Callable): the function to wrap an interface around. Often a machine learning model's prediction function.
        inputs (str | Component | List[str] | List[Component] | None): a single Gradio component, or list of Gradio components. Components can either be passed as instantiated objects, or referred to by their string shortcuts. The number of input components should match the number of parameters in fn. If set to None, then only the output components will be displayed.
        outputs (str | Component | List[str] | List[Component] | None): a single Gradio component, or list of Gradio components. Components can either be passed as instantiated objects, or referred to by their string shortcuts. The number of output components should match the number of values returned by fn. If set to None, then only the input components will be displayed.
        examples (List[List[Any]] | str | None): sample inputs for the function; if provided, appear below the UI components and can be clicked to populate the interface. Should be nested list, in which the outer list consists of samples and each inner list consists of an input corresponding to each input component. A string path to a directory of examples can also be provided. If there are multiple input components and a directory is provided, a log.csv file must be present in the directory to link corresponding inputs.
        cache_examples (bool | None): If True, caches examples in the server for fast runtime in examples. The default option in HuggingFace Spaces is True. The default option elsewhere is False.
        examples_per_page (int): If examples are provided, how many to display per page.
        live (bool): whether the interface should automatically rerun if any of the inputs change.
        interpretation (Callable | str): function that provides interpretation explaining prediction output. Pass "default" to use simple built-in interpreter, "shap" to use a built-in shapley-based interpreter, or your own custom interpretation function.
        num_shap (float): a multiplier that determines how many examples are computed for shap-based interpretation. Increasing this value will increase shap runtime, but improve results. Only applies if interpretation is "shap".
        title (str | None): a title for the interface; if provided, appears above the input and output components in large font.
        description (str | None): a description for the interface; if provided, appears above the input and output components and beneath the title in regular font. Accepts Markdown and HTML content.
        article (str | None): an expanded article explaining the interface; if provided, appears below the input and output components in regular font. Accepts Markdown and HTML content.
        thumbnail (str | None): path or url to image to use as display image when the web demo is shared on social media.
        theme (str | None): Theme to use - right now, only "default" is supported. Can be set with the GRADIO_THEME environment variable.
        css (str | None): custom css or path to custom css file to use with interface.
        allow_flagging (str | None): one of "never", "auto", or "manual". If "never" or "auto", users will not see a button to flag an input and output. If "manual", users will see a button to flag. If "auto", every prediction will be automatically flagged. If "manual", samples are flagged when the user clicks flag button. Can be set with environmental variable GRADIO_ALLOW_FLAGGING; otherwise defaults to "manual".
        flagging_options (List[str] | None): if provided, allows user to select from the list of options when flagging. Only applies if allow_flagging is "manual".
        flagging_dir (str): what to name the directory where flagged data is stored.
        flagging_callback (FlaggingCallback): An instance of a subclass of FlaggingCallback which will be called when a sample is flagged. By default logs to a local CSV file.
        analytics_enabled (bool | None): Whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable if defined, or default to True.
        """
        super().__init__(
            analytics_enabled=analytics_enabled, mode="interface", css=css, **kwargs
        )

        if inspect.iscoroutinefunction(fn):
            raise NotImplementedError(
                "Async functions are not currently supported within interfaces. Please use Blocks API."
            )
        self.interface_type = self.InterfaceTypes.STANDARD
        if (inputs is None or inputs == []) and (outputs is None or outputs == []):
            raise ValueError("Must provide at least one of `inputs` or `outputs`")
        elif outputs is None or outputs == []:
            outputs = []
            self.interface_type = self.InterfaceTypes.INPUT_ONLY
        elif inputs is None or inputs == []:
            inputs = []
            self.interface_type = self.InterfaceTypes.OUTPUT_ONLY

        if not isinstance(fn, list):
            fn = [fn]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        if self.is_space and cache_examples is None:
            self.cache_examples = True
        else:
            self.cache_examples = cache_examples or False

        if "state" in inputs or "state" in outputs:
            state_input_count = len([i for i in inputs if i == "state"])
            state_output_count = len([o for o in outputs if o == "state"])
            if state_input_count != 1 or state_output_count != 1:
                raise ValueError(
                    "If using 'state', there must be exactly one state input and one state output."
                )
            default = utils.get_default_args(fn[0])[inputs.index("state")]
            state_variable = Variable(value=default)
            inputs[inputs.index("state")] = state_variable
            outputs[outputs.index("state")] = state_variable

            if cache_examples:
                warnings.warn(
                    "Cache examples cannot be used with state inputs and outputs."
                    "Setting cache_examples to False."
                )
            self.cache_examples = False

        self.input_components = [
            get_component_instance(i, render=False) for i in inputs
        ]
        self.output_components = [
            get_component_instance(o, render=False) for o in outputs
        ]

        for component in self.input_components + self.output_components:
            if not (
                isinstance(component, IOComponent) or isinstance(component, Variable)
            ):
                raise ValueError(
                    f"{component} is not a valid input/output component for Interface."
                )

        if len(self.input_components) == len(self.output_components):
            same_components = [
                i is o for i, o in zip(self.input_components, self.output_components)
            ]
            if all(same_components):
                self.interface_type = self.InterfaceTypes.UNIFIED

        if self.interface_type in [
            self.InterfaceTypes.STANDARD,
            self.InterfaceTypes.OUTPUT_ONLY,
        ]:
            for o in self.output_components:
                o.interactive = False  # Force output components to be non-interactive

        if _repeat_outputs_per_model:
            self.output_components *= len(fn)

        if (
            interpretation is None
            or isinstance(interpretation, list)
            or callable(interpretation)
        ):
            self.interpretation = interpretation
        elif isinstance(interpretation, str):
            self.interpretation = [
                interpretation.lower() for _ in self.input_components
            ]
        else:
            raise ValueError("Invalid value for parameter: interpretation")

        self.api_mode = False
        self.predict = fn
        self.predict_durations = [[0, 0]] * len(fn)
        self.function_names = [func.__name__ for func in fn]
        self.__name__ = ", ".join(self.function_names)
        self.live = live
        self.title = title

        CLEANER = re.compile("<.*?>")

        def clean_html(raw_html):
            cleantext = re.sub(CLEANER, "", raw_html)
            return cleantext

        md = MarkdownIt(
            "js-default",
            {
                "linkify": True,
                "typographer": True,
                "html": True,
            },
        ).use(footnote_plugin)

        simple_description = None
        if description is not None:
            description = md.render(description)
            simple_description = clean_html(description)
        self.simple_description = simple_description
        self.description = description
        if article is not None:
            article = utils.readme_to_html(article)
            article = md.render(article)
        self.article = article

        self.thumbnail = thumbnail
        self.theme = theme or os.getenv("GRADIO_THEME", "default")
        if not (self.theme == "default"):
            warnings.warn("Currently, only the 'default' theme is supported.")

        if examples is None or (
            isinstance(examples, list)
            and (len(examples) == 0 or isinstance(examples[0], list))
        ):
            self.examples = examples
        elif (
            isinstance(examples, list) and len(self.input_components) == 1
        ):  # If there is only one input component, examples can be provided as a regular list instead of a list of lists
            self.examples = [[e] for e in examples]
        elif isinstance(examples, str):
            if not os.path.exists(examples):
                raise FileNotFoundError(
                    "Could not find examples directory: " + examples
                )
            log_file = os.path.join(examples, "log.csv")
            if not os.path.exists(log_file):
                if len(self.input_components) == 1:
                    exampleset = [
                        [os.path.join(examples, item)] for item in os.listdir(examples)
                    ]
                else:
                    raise FileNotFoundError(
                        "Could not find log file (required for multiple inputs): "
                        + log_file
                    )
            else:
                with open(log_file) as logs:
                    exampleset = list(csv.reader(logs))
                    exampleset = exampleset[1:]  # remove header
            for i, example in enumerate(exampleset):
                for j, (component, cell) in enumerate(
                    zip(
                        self.input_components + self.output_components,
                        example,
                    )
                ):
                    exampleset[i][j] = component.restore_flagged(
                        examples,
                        cell,
                        None,
                    )
            self.examples = exampleset
        else:
            raise ValueError(
                "Examples argument must either be a directory or a nested "
                "list, where each sublist represents a set of inputs."
            )
        self.num_shap = num_shap
        self.examples_per_page = examples_per_page

        self.simple_server = None

        # For analytics_enabled and allow_flagging: (1) first check for
        # parameter, (2) check for env variable, (3) default to True/"manual"
        self.analytics_enabled = (
            analytics_enabled
            if analytics_enabled is not None
            else os.getenv("GRADIO_ANALYTICS_ENABLED", "True") == "True"
        )
        if allow_flagging is None:
            allow_flagging = os.getenv("GRADIO_ALLOW_FLAGGING", "manual")
        if allow_flagging is True:
            warnings.warn(
                "The `allow_flagging` parameter in `Interface` now"
                "takes a string value ('auto', 'manual', or 'never')"
                ", not a boolean. Setting parameter to: 'manual'."
            )
            self.allow_flagging = "manual"
        elif allow_flagging == "manual":
            self.allow_flagging = "manual"
        elif allow_flagging is False:
            warnings.warn(
                "The `allow_flagging` parameter in `Interface` now"
                "takes a string value ('auto', 'manual', or 'never')"
                ", not a boolean. Setting parameter to: 'never'."
            )
            self.allow_flagging = "never"
        elif allow_flagging == "never":
            self.allow_flagging = "never"
        elif allow_flagging == "auto":
            self.allow_flagging = "auto"
        else:
            raise ValueError(
                "Invalid value for `allow_flagging` parameter."
                "Must be: 'auto', 'manual', or 'never'."
            )

        self.flagging_options = flagging_options
        self.flagging_callback = flagging_callback
        self.flagging_dir = flagging_dir

        self.save_to = None  # Used for selenium tests
        self.share = None
        self.share_url = None
        self.local_url = None

        self.requires_permissions = any(
            [component.requires_permissions for component in self.input_components]
        )

        self.favicon_path = None

        data = {
            "fn": fn,
            "inputs": inputs,
            "outputs": outputs,
            "live": live,
            "ip_address": self.ip_address,
            "interpretation": interpretation,
            "allow_flagging": allow_flagging,
            "custom_css": self.css is not None,
            "theme": self.theme,
        }

        if self.analytics_enabled:
            utils.initiated_analytics(data)

        utils.version_check()
        Interface.instances.add(self)

        param_names = inspect.getfullargspec(self.predict[0])[0]
        for component, param_name in zip(self.input_components, param_names):
            if component.label is None:
                component.label = param_name
        for i, component in enumerate(self.output_components):
            if component.label is None:
                if len(self.output_components) == 1:
                    component.label = "output"
                else:
                    component.label = "output " + str(i)

        if self.cache_examples and examples:
            cache_interface_examples(self)

        if self.allow_flagging != "never":
            if self.interface_type == self.InterfaceTypes.UNIFIED:
                self.flagging_callback.setup(self.input_components, self.flagging_dir)
            elif self.interface_type == self.InterfaceTypes.INPUT_ONLY:
                pass
            else:
                self.flagging_callback.setup(
                    self.input_components + self.output_components, self.flagging_dir
                )

        with self:
            if self.title:
                Markdown(
                    "<h1 style='text-align: center; margin-bottom: 1rem'>"
                    + self.title
                    + "</h1>"
                )
            if self.description:
                Markdown(self.description)

            def render_flag_btns(flagging_options):
                if flagging_options is None:
                    return [(Button("Flag"), None)]
                else:
                    return [
                        (
                            Button("Flag as " + flag_option),
                            flag_option,
                        )
                        for flag_option in flagging_options
                    ]

            with Row().style(equal_height=False):
                if self.interface_type in [
                    self.InterfaceTypes.STANDARD,
                    self.InterfaceTypes.INPUT_ONLY,
                    self.InterfaceTypes.UNIFIED,
                ]:
                    with Column(variant="panel"):
                        input_component_column = Column()
                        if self.interface_type in [
                            self.InterfaceTypes.INPUT_ONLY,
                            self.InterfaceTypes.UNIFIED,
                        ]:
                            status_tracker = StatusTracker(cover_container=True)
                        with input_component_column:
                            for component in self.input_components:
                                component.render()
                        if self.interpretation:
                            interpret_component_column = Column(visible=False)
                            interpretation_set = []
                            with interpret_component_column:
                                for component in self.input_components:
                                    interpretation_set.append(Interpretation(component))
                        with Row().style(mobile_collapse=False):
                            if self.interface_type in [
                                self.InterfaceTypes.STANDARD,
                                self.InterfaceTypes.INPUT_ONLY,
                            ]:
                                clear_btn = Button("Clear")
                                if not self.live:
                                    submit_btn = Button("Submit", variant="primary")
                            elif self.interface_type == self.InterfaceTypes.UNIFIED:
                                clear_btn = Button("Clear")
                                submit_btn = Button("Submit", variant="primary")
                                if self.allow_flagging == "manual":
                                    flag_btns = render_flag_btns(self.flagging_options)

                if self.interface_type in [
                    self.InterfaceTypes.STANDARD,
                    self.InterfaceTypes.OUTPUT_ONLY,
                ]:

                    with Column(variant="panel"):
                        status_tracker = StatusTracker(cover_container=True)
                        for component in self.output_components:
                            component.render()
                        with Row().style(mobile_collapse=False):
                            if self.interface_type == self.InterfaceTypes.OUTPUT_ONLY:
                                clear_btn = Button("Clear")
                                submit_btn = Button("Generate", variant="primary")
                            if self.allow_flagging == "manual":
                                flag_btns = render_flag_btns(self.flagging_options)
                            if self.interpretation:
                                interpretation_btn = Button("Interpret")

            async def submit_fn(*args):
                if len(self.output_components) == 1:
                    return (await self.run_prediction(args))[0]
                else:
                    return await self.run_prediction(args)

            if self.live:
                if self.interface_type == self.InterfaceTypes.OUTPUT_ONLY:
                    super().load(submit_fn, None, self.output_components)
                    submit_btn.click(
                        submit_fn,
                        None,
                        self.output_components,
                        api_name="predict",
                        status_tracker=status_tracker,
                    )
                else:
                    for component in self.input_components:
                        if isinstance(component, Streamable):
                            if component.streaming:
                                component.stream(
                                    submit_fn,
                                    self.input_components,
                                    self.output_components,
                                )
                                continue
                            else:
                                print(
                                    "Hint: Set streaming=True for "
                                    + component.__class__.__name__
                                    + " component to use live streaming."
                                )
                        if isinstance(component, Changeable):
                            component.change(
                                submit_fn, self.input_components, self.output_components
                            )
            else:
                submit_btn.click(
                    submit_fn,
                    self.input_components,
                    self.output_components,
                    api_name="predict",
                    scroll_to_output=True,
                    status_tracker=status_tracker,
                )
            clear_btn.click(
                None,
                [],
                (
                    self.input_components
                    + self.output_components
                    + (
                        [input_component_column]
                        if self.interface_type
                        in [
                            self.InterfaceTypes.STANDARD,
                            self.InterfaceTypes.INPUT_ONLY,
                            self.InterfaceTypes.UNIFIED,
                        ]
                        else []
                    )
                    + ([interpret_component_column] if self.interpretation else [])
                ),
                _js=f"""() => {json.dumps(
                    [component.cleared_value if hasattr(component, "cleared_value") else None
                    for component in self.input_components + self.output_components] + (
                            [Column.update(visible=True)]
                            if self.interface_type
                            in [
                                self.InterfaceTypes.STANDARD,
                                self.InterfaceTypes.INPUT_ONLY,
                                self.InterfaceTypes.UNIFIED,
                            ]
                            else []
                        )
                    + ([Column.update(visible=False)] if self.interpretation else [])
                )}
                """,
            )

            class FlagMethod:
                def __init__(self, flagging_callback, flag_option=None):
                    self.flagging_callback = flagging_callback
                    self.flag_option = flag_option

                def __call__(self, *flag_data):
                    self.flagging_callback.flag(flag_data, flag_option=self.flag_option)

            if self.allow_flagging == "manual":
                if self.interface_type in [
                    self.InterfaceTypes.STANDARD,
                    self.InterfaceTypes.OUTPUT_ONLY,
                    self.InterfaceTypes.UNIFIED,
                ]:
                    if self.interface_type == self.InterfaceTypes.UNIFIED:
                        flag_components = self.input_components
                    else:
                        flag_components = self.input_components + self.output_components
                    for flag_btn, flag_option in flag_btns:
                        flag_method = FlagMethod(self.flagging_callback, flag_option)
                        flag_btn.click(
                            flag_method,
                            inputs=flag_components,
                            outputs=[],
                            _preprocess=False,
                            queue=False,
                        )

            if self.examples:
                non_state_inputs = [
                    c for c in self.input_components if not isinstance(c, Variable)
                ]

                examples = Dataset(
                    components=non_state_inputs,
                    samples=self.examples,
                    type="index",
                )

                def load_example(example_id):
                    processed_examples = [
                        component.preprocess_example(sample)
                        for component, sample in zip(
                            self.input_components, self.examples[example_id]
                        )
                    ]
                    if self.cache_examples:
                        processed_examples += load_from_cache(self, example_id)
                    if len(processed_examples) == 1:
                        return processed_examples[0]
                    else:
                        return processed_examples

                examples.click(
                    load_example,
                    inputs=[examples],
                    outputs=non_state_inputs
                    + (self.output_components if self.cache_examples else []),
                    _postprocess=False,
                    queue=False,
                )

            if self.interpretation:
                interpretation_btn.click(
                    lambda *data: self.interpret(data)
                    + [Column.update(visible=False), Column.update(visible=True)],
                    inputs=self.input_components + self.output_components,
                    outputs=interpretation_set
                    + [input_component_column, interpret_component_column],
                    status_tracker=status_tracker,
                    _preprocess=False,
                )

            if self.article:
                Markdown(self.article)

        self.config = self.get_config_file()

    def __call__(self, *params):
        if (
            self.api_mode
        ):  # skip the preprocessing/postprocessing if sending to a remote API
            output = self.run_prediction(params, called_directly=True)
        else:
            output = self.process(params)
        return output[0] if len(output) == 1 else output

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr = "Gradio Interface for: {}".format(
            ", ".join(fn.__name__ for fn in self.predict)
        )
        repr += "\n" + "-" * len(repr)
        repr += "\ninputs:"
        for component in self.input_components:
            repr += "\n|-{}".format(str(component))
        repr += "\noutputs:"
        for component in self.output_components:
            repr += "\n|-{}".format(str(component))
        return repr

    async def run_prediction(
        self,
        processed_input: List[Any],
        called_directly: bool = False,
    ) -> List[Any] | Tuple[List[Any], List[float]]:
        """
        Runs the prediction function with the given (already processed) inputs.
        Parameters:
        processed_input (list): A list of processed inputs.
        called_directly (bool): Whether the prediction is being called
            directly (i.e. as a function, not through the GUI).
        Returns:
        predictions (list): A list of predictions (not post-processed).
        """
        if self.api_mode:  # Serialize the input
            processed_input = [
                input_component.serialize(processed_input[i], called_directly)
                for i, input_component in enumerate(self.input_components)
            ]
        predictions = []
        output_component_counter = 0

        for predict_fn in self.predict:
            prediction = predict_fn(*processed_input)
            if isinstance(prediction, list) or isinstance(prediction, tuple):
                predictions.extend(prediction)
            else:
                predictions.append(prediction)
            

        async_predictions = [pred for pred in predictions if isinstance(pred, asyncio.Future)]
        sync_predictions = [pred for pred in predictions if not isinstance(pred, asyncio.Future)]

        async_predictions = await asyncio.gather(*async_predictions)
        results = sync_predictions
        for pred in async_predictions:
            if isinstance(pred, tuple):
                results.extend(pred)
            else:
                results.append(pred)
        return results

    def process(self, raw_input: List[Any]) -> Tuple[List[Any], List[float]]:
        """
        First preprocesses the input, then runs prediction using
        self.run_prediction(), then postprocesses the output.
        Parameters:
        raw_input: a list of raw inputs to process and apply the prediction(s) on.
        Returns:
        processed output: a list of processed  outputs to return as the prediction(s).
        duration: a list of time deltas measuring inference time for each prediction fn.
        """
        processed_input = [
            input_component.preprocess(raw_input[i])
            for i, input_component in enumerate(self.input_components)
        ]
        predictions = self.run_prediction(processed_input)
        processed_output = [
            output_component.postprocess(predictions[i])
            if predictions[i] is not None
            else None
            for i, output_component in enumerate(self.output_components)
        ]
        return processed_output

    def interpret(self, raw_input: List[Any]) -> List[Any]:
        return [
            {"original": raw_value, "interpretation": interpretation}
            for interpretation, raw_value in zip(
                interpretation.run_interpret(self, raw_input)[0], raw_input
            )
        ]

    def test_launch(self) -> None:
        """
        Passes a few samples through the function to test if the inputs/outputs
        components are consistent with the function parameter and return values.
        """
        for predict_fn in self.predict:
            print("Test launch: {}()...".format(predict_fn.__name__), end=" ")
            raw_input = []
            for input_component in self.input_components:
                if input_component.test_input is None:
                    print("SKIPPED")
                    break
                else:
                    raw_input.append(input_component.test_input)
            else:
                self.process(raw_input)
                print("PASSED")
                continue

    def integrate(self, comet_ml=None, wandb=None, mlflow=None) -> None:
        """
        A catch-all method for integrating with other libraries.
        Should be run after launch()
        Parameters:
            comet_ml (Experiment): If a comet_ml Experiment object is provided,
            will integrate with the experiment and appear on Comet dashboard
            wandb (module): If the wandb module is provided, will integrate
            with it and appear on WandB dashboard
            mlflow (module): If the mlflow module  is provided, will integrate
            with the experiment and appear on ML Flow dashboard
        """
        analytics_integration = ""
        if comet_ml is not None:
            analytics_integration = "CometML"
            comet_ml.log_other("Created from", "Gradio")
            if self.share_url is not None:
                comet_ml.log_text("gradio: " + self.share_url)
                comet_ml.end()
            else:
                comet_ml.log_text("gradio: " + self.local_url)
                comet_ml.end()
        if wandb is not None:
            analytics_integration = "WandB"
            if self.share_url is not None:
                wandb.log(
                    {
                        "Gradio panel": wandb.Html(
                            '<iframe src="'
                            + self.share_url
                            + '" width="'
                            + str(self.width)
                            + '" height="'
                            + str(self.height)
                            + '" frameBorder="0"></iframe>'
                        )
                    }
                )
            else:
                print(
                    "The WandB integration requires you to "
                    "`launch(share=True)` first."
                )
        if mlflow is not None:
            analytics_integration = "MLFlow"
            if self.share_url is not None:
                mlflow.log_param("Gradio Interface Share Link", self.share_url)
            else:
                mlflow.log_param("Gradio Interface Local Link", self.local_url)
        if self.analytics_enabled and analytics_integration:
            data = {"integration": analytics_integration}
            utils.integration_analytics(data)


class TabbedInterface(Blocks):
    """
    A TabbedInterface is created by providing a list of Interfaces, each of which gets
    rendered in a separate tab.
    """

    def __init__(
        self, interface_list: List[Interface], tab_names: Optional[List[str]] = None
    ):
        """
        Parameters:
        interface_list (List[Interface]): a list of interfaces to be rendered in tabs.
        tab_names (List[str] | None): a list of tab names. If None, the tab names will be "Tab 1", "Tab 2", etc.
        Returns:
        (gradio.TabbedInterface): a Gradio Tabbed Interface for the given interfaces
        """
        if tab_names is None:
            tab_names = ["Tab {}".format(i) for i in range(len(interface_list))]
        super().__init__()
        with self:
            with Tabs():
                for (interface, tab_name) in zip(interface_list, tab_names):
                    with TabItem(label=tab_name):
                        interface.render()


def close_all(verbose: bool = True) -> None:
    for io in Interface.get_instances():
        io.close(verbose)
