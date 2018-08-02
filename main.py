import numpy as np

from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, \
    ColumnDataSource, PrintfTickFormatter, HoverTool
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models.widgets import Button, Slider, Select, Panel, \
    Tabs, CheckboxGroup, TextInput, Div, Toggle

from simceo import broker

print("*********************************")
print("**   STARTING SIMCEO SERVER    **")
print("*********************************")
agent = broker()
agent.start()

doc = curdoc()

## INFO
info = Div(text="""<h2>Time: 0.0s</h2>""")

## OPS
select = Select(title="Optical Path:", value="", options=[""])

## REFRESH
refresh = Button(label="REFRESH", button_type="success")
def cb_refresh():
    if agent.ops:
        options  = [op.sensor_class for op in agent.ops]
        select.options = options
        select.value = options[0]
refresh.on_click(cb_refresh)

## STREAM
stream = Toggle(label='STREAM',button_type="default",active=False)

## PUPIL PLANE WAVEFRONT
NPX = 512
W = ColumnDataSource(data=dict(W=[np.zeros((NPX,NPX))]))
L = 25.5
ppw = figure(plot_width=450,plot_height=400,x_range=[-L/2,L/2],y_range=[-L/2,L/2])
cmpr = LinearColorMapper(palette="Viridis256")
ppw.image('W',source=W, x=-L/2, y=-L/2, dw=L ,dh=L, color_mapper=cmpr)
#cb = ColorBar(color_mapper=cmpr,location=(-40,0),label_standoff=10,
#              formatter=PrintfTickFormatter(format="%.2f"),width=70)
#ppw.add_layout(cb,'left')
def update():
    #print("update")
    if agent.ops:
        info.text="""<h2>Time: {0:.3f}s</h2>""".format(agent.currentTime)
        op = agent.ops[select.options.index(select.value)]
        #print(op.src.wavefront.rms(-9))
        W.data.update(dict(W=[op.src.phase.host(units='nm')]))
    else:
        stream.active = False
        select.options = [""]
        select.value = ""

callback_id = None
def cb_stream(attrname, old, new):
    global callback_id
    print(attrname, old, new)
    if new:
        print('Add callback')
        callback_id = doc.add_periodic_callback(update,2000)
    else:
        print('Remove callback')
        doc.remove_periodic_callback(callback_id)
stream.on_change('active',cb_stream)

doc.add_root(row(widgetbox([info,refresh,select,stream]),ppw))
