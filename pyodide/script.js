importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.3/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.1/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.1/dist/wheels/panel-0.14.1-py3-none-any.whl', 'pyodide-http==0.1.0', 'holoviews>=1.15.1', 'holoviews>=1.15.1', 'hvplot', 'numpy', 'pandas', 'param', 'scikit-learn', 'xgboost']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import panel as pn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import param
import holoviews as hv

import pandas as pd
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv
from holoviews.streams import Buffer
from bokeh.models import Button, Slider, Spinner
import time
import asyncio
pn.extension('katex')
pn.extension(sizing_mode="stretch_width", template="fast")
pn.state.template.param.update(title="TTnet Vizualisation")
pipeline = pn.pipeline.Pipeline()
iris_df = load_iris(as_frame=True)
pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')
pn.extension(sizing_mode="stretch_width")

class Stage1(param.Parameterized):
    @param.output(('dataset_value', param.String), ('ksize_value', param.Number), ('filtersize_value', param.Number), ('stride_value', param.Number))
    def output(self):
        return self.dataset_value, self.ksize_value, self.filtersize_value, self.stride_value

    def pipeline1(self, dataset, ksize, filtersize, stride):
        self.dataset_value = dataset
        self.ksize_value = ksize
        self.filtersize_value = filtersize
        self.stride_value = stride
        return None #pn.pane.LaTeX(dataset,style={'font-size': '2em'})
    def panel(self):
        txt0 = pn.pane.LaTeX("Dataset", style={'font-size': '2em'})
        dataset = pn.widgets.Select(name='Proposed Datasets', options=['Adult', 'Diabetes'], size=2)
        txt = pn.pane.LaTeX("TTnet Hyperparameters", style={'font-size': '2em'})
        txt2 = pn.pane.LaTeX("", style={'font-size': '2em'})
        ksize = pn.widgets.IntSlider(start=5, end=30, name="Kernel size")
        filtersize = pn.widgets.IntSlider(start=2, end=30, name="Number of filter")
        stride = pn.widgets.IntSlider(start=2, end=30, name="Stride Size")
        return pn.Column(txt0, txt2, dataset, txt2, txt, txt2, ksize, filtersize, stride, pn.bind(self.pipeline1,  dataset, ksize, filtersize, stride))

class FakeInstrument(object):
    def __init__(self, offset=0.0):
        self.offset = offset

    def set_offset(self, value):
        self.offset = value

    def read_data(self):
        return np.random.random() + self.offset

instrument = FakeInstrument()
def make_df(time_sec=0.0, temperature_degC=0.0):
    return pd.DataFrame({' Epochs (s)': 10*time_sec, ' Loss (MSE)': temperature_degC}, index=[0])

example_df = pd.DataFrame(columns=make_df().columns)
buffer = Buffer(example_df, length=1000, index=False)
plot = hv.DynamicMap(hv.Curve, streams=[buffer]).opts(padding=0.1, height=300, xlim=(0, None), responsive=True)
LABEL_START = 'Start'
LABEL_STOP = 'Stop'
LABEL_CSV_START = "Save to csv"
LABEL_CSV_STOP = "Stop save"
CSV_FILENAME = 'tmp.csv'

button_startstop = Button(label=LABEL_START, button_type="primary")
button_csv = Button(label=LABEL_CSV_START, button_type="success")
offset = Slider(title='Offset', start=-10.0, end=10.0, value=0.0, step=0.1)
interval = Spinner(title="Interval (sec)", value=0.1, step=0.01)

acquisition_task = None
save_to_csv = False

async def acquire_data(interval_sec=0.1):
    global save_to_csv, print_acc
    print_acc = False
    t0 = time.time()
    for epoch in range(10):
        instrument.set_offset(offset.value)
        time_elapsed = time.time() - t0
        value = instrument.read_data()
        b = make_df(time_elapsed, value)
        buffer.send(b)

        if save_to_csv:
            b.to_csv(CSV_FILENAME, header=False, index=False, mode='a')

        time_spent_buffering = time.time() - t0 - time_elapsed
        if interval_sec > time_spent_buffering:
            await asyncio.sleep(interval_sec - time_spent_buffering)
    if epoch==9:
        print_acc = True

def toggle_csv(*events):
    global save_to_csv
    pn.WidgetBox('# VIN')
    """if button_csv.label == LABEL_CSV_START:
        button_csv.label = LABEL_CSV_STOP
        example_df.to_csv(CSV_FILENAME, index=False)  # example_df is empty, so this just writes the header
        save_to_csv = True
    else:
        save_to_csv = False
        button_csv.label = LABEL_CSV_START"""


def start_stop(*events):
    global acquisition_task, save_to_csv, print_acc
    if button_startstop.label == LABEL_START:
        button_startstop.label = LABEL_STOP
        buffer.clear()
        acquisition_task = asyncio.get_running_loop().create_task(acquire_data(interval_sec=interval.value))
        if print_acc:
            toggle_csv()


    else:
        acquisition_task.cancel()
        button_startstop.label = LABEL_START
        #if save_to_csv:
        #    toggle_csv()


button_startstop.on_click(start_stop)
button_csv.on_click(toggle_csv)

button = pn.widgets.Button(name='Test Accuracy', button_type='primary')
text = pn.widgets.TextInput(value='-')

#pn.Row(button, text)

hv.extension('bokeh')
hv.renderer('bokeh').theme = 'caliber'
controls = pn.WidgetBox('# Train',
                        button_startstop,
                        #button_csv,
                        button,
                        text,
                        )

app = pn.Row(plot, controls)


class Stage2(param.Parameterized):
    dataset_value = param.String()
    ksize_value = param.Number()
    filtersize_value = param.Number()
    stride_value = param.Number()
    @param.output(('dataset_value', param.String), ('ksize_value', param.Number), ('filtersize_value', param.Number),
                  ('stride_value', param.Number), ('accuracy', param.Number))
    def output(self):
        return self.dataset_value, self.ksize_value, self.filtersize_value, self.stride_value, self.accuracy
    def pipeline2(self, dataset, ksize, filter, stride):
        model = XGBClassifier(max_depth=int(filter), n_estimators=int(ksize))
        model.fit(iris_df.data, iris_df.target)
        accuracy = round(accuracy_score(iris_df.target, model.predict(iris_df.data)) * 100, 1)
        self.accuracy = accuracy

        def b(event):
            text.value = '{0}%'.format(accuracy)

        button.on_click(b)

        return None

    @param.depends('dataset_value', 'ksize_value', 'filtersize_value', 'stride_value')
    def panel(self):
        info0 = pn.pane.LaTeX('Dataset used: '+self.dataset_value ,
                      style={'font-size': '2em'})
        info1 = pn.pane.LaTeX(
            '',
            style={'font-size': '2em'})
        info2 = pn.pane.LaTeX(' Hyperparameters: ksize ' + str(self.ksize_value) +
            ', filters ' + str(self.filtersize_value) + ', stride ' + str(self.stride_value),
            style={'font-size': '2em'})
        app1 = pn.Column(info0, info1, info2, info1)

        app3 = pn.bind(self.pipeline2, self.dataset_value, self.ksize_value, self.filtersize_value,
                str(self.stride_value))

        return pn.Column(app1, app, app3)




class Stage3(param.Parameterized):
    dataset_value = param.String()
    ksize_value = param.Number()
    filtersize_value = param.Number()
    stride_value = param.Number()
    accuracy = param.Number()

    @param.depends('dataset_value', 'ksize_value', 'filtersize_value', 'stride_value', 'accuracy')
    def panel(self):
        info0 = pn.pane.LaTeX('Dataset used: ' + self.dataset_value,
                              style={'font-size': '2em'})
        info1 = pn.pane.LaTeX(
            '',
            style={'font-size': '2em'})
        info2 = pn.pane.LaTeX(' Hyperparameters: ksize ' + str(self.ksize_value) +
                              ', filters ' + str(self.filtersize_value) + ', stride ' + str(self.stride_value),
                              style={'font-size': '2em'})

        info3 = pn.pane.LaTeX(
            'Test accuracy: ' + str(self.accuracy) + '%',
            style={'font-size': '2em'})
        info4 = pn.pane.LaTeX(
            'Rules Select ',
            style={'font-size': '2em'})
        info5 = pn.pane.LaTeX(
            'Rules: CNF/DNF format: ',
            style={'font-size': '2em'})
        x = pn.widgets.Select(name='Rule to plot', options=["Rule 1", "Rule 2"], value='Rule 1')
        change_buton = Button(label="Info rule", button_type="primary")
        #y = pn.widgets.Select(name='Format', options=["CNF", "DNF", "Graph"], value='Graph')
        textCNF = pn.widgets.TextInput(value='-')
        textDNF = pn.widgets.TextInput(value='-')


        def c(event):
            if x.value == "Rule 1":
                textCNF.value = "x1 & x2"
                textDNF.value = "x1 & x2"
            elif x.value == "Rule 2":
                textCNF.value = "~x1 & ~x2"
                textDNF.value = "~x1 & ~x2"
            #text.value = '{0}%'.format(accuracy)

        change_buton.on_click(c)






        return pn.Column(info0, info1, info2, info1, info3, info1, info4, x, change_buton, info1, textCNF, textDNF )




pipeline = pn.pipeline.Pipeline(debug=True, show_header=False)
#stage1 = Stage1()
pipeline.add_stage('Data&Model', Stage1())
#stage2 = Stage2(dataset_value=stage1.output()[0], ksize_value=stage1.output()[1], filtersize_value=stage1.output()[2], stride_value=stage1.output()[3])
pipeline.add_stage('Train', Stage2())
pipeline.add_stage('Rules', Stage3())
pn.Column(pipeline).servable()


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()