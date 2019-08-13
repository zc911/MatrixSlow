from pyecharts import options as opts
from pyecharts.charts import Graph

from ..core import default_graph


def draw_graph():
    nodes_for_draw = []
    links_for_draw = []
    for node in default_graph.nodes:
        nodes_for_draw.append({'name': node.name, "symbolSize": 50})
    for node in default_graph.nodes:
        for child in node.children:
            links_for_draw.append({'source': node.name, 'target': child.name})
    graph = Graph(init_opts=opts.InitOpts(width='1800px', height='1000px'))
    graph.set_global_opts(
        title_opts=opts.TitleOpts(title="MatrixSlow Graph"))
    graph.add("", nodes_for_draw, links_for_draw, layout='force',
              repulsion=8000, edge_symbol=['circle', 'arrow'])

    graph.render('./aaa.html')
