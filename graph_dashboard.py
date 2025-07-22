
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from graph_builder import DarkSoulsGraphBuilder
import numpy as np

st.set_page_config(page_title="Dark Souls Knowledge Graph", layout="wide")

@st.cache_data
def load_graph_builder():
    """Load graph builder with caching"""
    return DarkSoulsGraphBuilder()

def create_network_plot(graph, sample_size=200):
    """Create interactive network plot"""
    if len(graph.nodes()) > sample_size:
        nodes_to_include = list(graph.nodes())[:sample_size]
        subgraph = graph.subgraph(nodes_to_include)
    else:
        subgraph = graph
    
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    # Prepare node data
    node_x = [pos[node][0] for node in subgraph.nodes()]
    node_y = [pos[node][1] for node in subgraph.nodes()]
    node_colors = [subgraph.nodes[node].get('color', '#DDA0DD') for node in subgraph.nodes()]
    node_sizes = [subgraph.degree(node) * 10 + 10 for node in subgraph.nodes()]
    node_text = [f"{node}<br>Type: {subgraph.nodes[node].get('type', 'Unknown')}<br>"
                f"Category: {subgraph.nodes[node].get('category', 'Unknown')}<br>"
                f"Connections: {subgraph.degree(node)}" 
                for node in subgraph.nodes()]
    
    # Prepare edge data
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create traces
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                           hoverinfo='none', mode='lines')
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                           text=node_text, marker=dict(size=node_sizes, color=node_colors,
                                                      line=dict(width=2, color='white')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title='Dark Souls Knowledge Graph',
                                  showlegend=False, hovermode='closest',
                                  margin=dict(b=20,l=5,r=5,t=40),
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig

def main():
    st.title("🎮 Dark Souls Knowledge Graph Dashboard")
    
    # Load data
    with st.spinner("Loading graph data..."):
        builder = load_graph_builder()
    
    # Sidebar controls
    st.sidebar.header("Graph Controls")
    
    # Display options
    view_type = st.sidebar.selectbox("View Type", 
                                   ["Full Graph", "Category Filter", "Entity Search", "Statistics"])
    
    if view_type == "Full Graph":
        st.header("Complete Knowledge Graph")
        sample_size = st.sidebar.slider("Sample Size", 50, 1000, 300)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_network_plot(builder.graph, sample_size)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            stats = builder.get_graph_statistics()
            st.subheader("Graph Statistics")
            st.metric("Total Nodes", stats['nodes'])
            st.metric("Total Edges", stats['edges'])
            st.metric("Connected Components", stats['connected_components'])
            st.metric("Graph Density", f"{stats['density']:.6f}")
            
            st.subheader("Entity Types")
            for entity_type, count in stats['node_types'].items():
                st.write(f"**{entity_type}**: {count}")
    
    elif view_type == "Category Filter":
        st.header("Category-based Exploration")
        
        categories = builder.entities_df['category'].unique()
        selected_category = st.sidebar.selectbox("Select Category", categories)
        
        # Filter entities by category
        category_entities = builder.entities_df[builder.entities_df['category'] == selected_category]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{selected_category} Entities")
            
            # Create subgraph for this category
            category_nodes = category_entities['entity'].tolist()[:100]  # Limit for performance
            if category_nodes:
                subgraph = builder.graph.subgraph(category_nodes)
                if subgraph.number_of_edges() > 0:
                    fig = create_network_plot(subgraph, len(category_nodes))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No connections found between entities in this category")
            
        with col2:
            st.subheader("Category Details")
            st.metric("Total Entities", len(category_entities))
            
            # Show entity types in this category
            entity_types = category_entities['type'].value_counts()
            st.write("**Entity Types:**")
            for etype, count in entity_types.items():
                st.write(f"- {etype}: {count}")
            
            # Show subcategories
            subcategories = category_entities['subcategory'].value_counts()
            if len(subcategories) > 1:
                st.write("**Subcategories:**")
                for subcat, count in subcategories.head(10).items():
                    st.write(f"- {subcat}: {count}")
    
    elif view_type == "Entity Search":
        st.header("Entity Explorer")
        
        # Search box
        search_term = st.text_input("Search for entities:", "")
        
        if search_term:
            # Find matching entities
            matching_entities = builder.entities_df[
                builder.entities_df['entity'].str.contains(search_term, case=False, na=False)
            ]
            
            if len(matching_entities) > 0:
                st.subheader(f"Found {len(matching_entities)} matching entities:")
                
                # Display results
                for _, entity in matching_entities.head(20).iterrows():
                    with st.expander(f"{entity['entity']} ({entity['type']})"):
                        st.write(f"**Category:** {entity['category']}")
                        st.write(f"**Subcategory:** {entity['subcategory']}")
                        st.write(f"**Type:** {entity['type']}")
                        
                        # Show connections if entity is in graph
                        if entity['entity'] in builder.graph.nodes:
                            neighbors = list(builder.graph.neighbors(entity['entity']))
                            st.write(f"**Connections ({len(neighbors)}):** {', '.join(neighbors[:10])}")
                            if len(neighbors) > 10:
                                st.write(f"... and {len(neighbors) - 10} more")
            else:
                st.info("No entities found matching your search term")
    
    elif view_type == "Statistics":
        st.header("Detailed Graph Analysis")
        
        stats = builder.get_graph_statistics()
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", stats['nodes'])
        with col2:
            st.metric("Edges", stats['edges'])
        with col3:
            st.metric("Components", stats['connected_components'])
        with col4:
            st.metric("Density", f"{stats['density']:.6f}")
        
        # Entity type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entity Types Distribution")
            entity_type_df = pd.DataFrame(list(stats['node_types'].items()), 
                                        columns=['Type', 'Count'])
            fig = px.pie(entity_type_df, values='Count', names='Type', 
                        title="Entity Types")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Category Distribution")
            category_counts = builder.entities_df['category'].value_counts().head(10)
            fig = px.bar(x=category_counts.index, y=category_counts.values,
                        title="Top 10 Categories")
            fig.update_xaxes(title="Category")
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Degree distribution
        st.subheader("Node Degree Distribution")
        degrees = [builder.graph.degree(node) for node in builder.graph.nodes()]
        fig = px.histogram(x=degrees, nbins=20, title="Node Degree Distribution")
        fig.update_xaxes(title="Degree")
        fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        # Most connected entities
        st.subheader("Most Connected Entities")
        degree_centrality = nx.degree_centrality(builder.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        
        top_nodes_df = pd.DataFrame(top_nodes, columns=['Entity', 'Degree Centrality'])
        st.dataframe(top_nodes_df, use_container_width=True)

if __name__ == "__main__":
    main()
