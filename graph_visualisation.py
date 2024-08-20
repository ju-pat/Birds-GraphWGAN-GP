import networkx as nx
import matplotlib.pyplot as plt
import folium
import random

def visualize_graph(data, dirige=True):
    if dirige:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for i, (lon, lat) in enumerate(data.x[:, :2]):
        G.add_node(i, pos=(lon.item(), lat.item()))
    for edge in data.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    pos = nx.get_node_attributes(G, "pos")
    # nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", font_size=8)

    if dirige:
        # Draw directed graph
        nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=16, font_weight='bold', arrows=True)
    else:
        # Draw undirected graph
        nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=16, font_weight='bold')

    first_point = (data.x[0][0].item(), data.x[0][1].item())
    last_point = (data.x[-1][0].item(), data.x[-1][1].item())
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=100, node_color="red")
    nx.draw_networkx_nodes(G, pos, nodelist=[len(data.x)-1], node_size=100, node_color="red")

    plt.title("Graph Visualization")
    plt.show()

def visualize_and_save_graph(data, dir):
    G = nx.Graph()
    for i, (lon, lat) in enumerate(data.x):
        G.add_node(i, pos=(lon.item(), lat.item()))
    for edge in data.edge_index.t().tolist():
        G.add_edge(*edge)

    pos = nx.get_node_attributes(G, "pos")
    nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", font_size=8)

    first_point = (data.x[0][0].item(), data.x[0][1].item())
    last_point = (data.x[-1][0].item(), data.x[-1][1].item())
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=100, node_color="red")
    nx.draw_networkx_nodes(G, pos, nodelist=[len(data.x)-1], node_size=100, node_color="red")

    plt.title("Graph Visualization")
    plt.savefig(dir+'.png')
    plt.show()


# Plot the map with the trajectory

def random_color():
    r =lambda: random.randint(0,255)
    return('#%02X%02X%02X' % (r(),r(),r()))

def plot_map(df_data, colony, save = False, path = "") :
    m = folium.Map(location = colony, zoom_start=8)

    folium.Marker(colony, popup="<i>Colony</i>").add_to(m)
    i = 1
    for df in df_data:
        t = f"Trajectory {i}"
        locations = list(zip(df['lat'], df['lon']))
        
        folium.PolyLine(tooltip = t,
                    locations = locations,
                    color = random_color(),
                    weight=2,
                    opacity=1).add_to(m)
        i += 1
        
    display(m) 
    
    if save : 
        m.save(path)       
    return
