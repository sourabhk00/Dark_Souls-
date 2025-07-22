
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DarkSoulsGraphBuilder:
    def __init__(self, entities_csv="knowledge_graph/entities.csv", relationships_csv="knowledge_graph/relationships.csv"):
        """Initialize the graph builder with data files"""
        self.entities_df = pd.read_csv(entities_csv)
        self.relationships_df = pd.read_csv(relationships_csv)
        self.graph = None
        
        # Color schemes for different entity types
        self.entity_colors = {
            'PERSON': '#FF6B6B',      # Red for characters
            'ORG': '#4ECDC4',         # Teal for organizations  
            'WORK_OF_ART': '#45B7D1', # Blue for items/weapons
            'GPE': '#96CEB4',         # Green for locations
            'LOC': '#FFEAA7',         # Yellow for specific places
            'UNKNOWN': '#DDA0DD'      # Purple for unknown
        }
        
        self.category_colors = {
            'Weapons': '#FF4757',
            'Shields': '#3742FA',
            'Consumables': '#2ED573',
            'Bosses': '#FF6348',
            'Enemies': '#FF7675',
            'NPCs': '#74B9FF',
            'Areas': '#00B894',
            'Rings': '#FDCB6E',
            'Spells': '#6C5CE7'
        }
        
        print(f"📊 Loaded {len(self.entities_df)} entities and {len(self.relationships_df)} relationships")
        self._build_graph()
    
    def _build_graph(self):
        """Build the NetworkX graph from the data"""
        self.graph = nx.Graph()
        
        # Add nodes with attributes
        for _, entity in self.entities_df.iterrows():
            self.graph.add_node(
                entity['entity'],
                type=entity['type'],
                category=entity['category'],
                subcategory=entity['subcategory'],
                color=self.entity_colors.get(entity['type'], self.entity_colors['UNKNOWN'])
            )
        
        # Add edges
        for _, rel in self.relationships_df.iterrows():
            if rel['subject'] in self.graph.nodes and rel['object'] in self.graph.nodes:
                self.graph.add_edge(
                    rel['subject'],
                    rel['object'],
                    predicate=rel['predicate'],
                    rel_type=rel['type'],
                    category=rel['category']
                )
        
        print(f"🔗 Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_graph_statistics(self):
        """Get comprehensive graph statistics"""
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'largest_component_size': len(max(nx.connected_components(self.graph), key=len)),
        }
        
        # Node statistics by type
        node_types = [self.graph.nodes[node].get('type', 'UNKNOWN') for node in self.graph.nodes()]
        stats['node_types'] = Counter(node_types)
        
        # Degree distribution
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        stats['degree_stats'] = {
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'max': max(degrees),
            'min': min(degrees)
        }
        
        return stats
    
    def create_basic_visualization(self, layout='spring', node_size_factor=100, figsize=(15, 10)):
        """Create basic matplotlib visualization"""
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.random_layout(self.graph)
        
        # Get node colors and sizes
        node_colors = [self.graph.nodes[node].get('color', '#DDA0DD') for node in self.graph.nodes()]
        node_sizes = [self.graph.degree(node) * node_size_factor + 50 for node in self.graph.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray', width=0.5)
        
        # Add labels for high-degree nodes only
        high_degree_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) > 2]
        labels = {node: node for node in high_degree_nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold')
        
        plt.title("Dark Souls Knowledge Graph", size=16, fontweight='bold')
        plt.axis('off')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=entity_type)
                          for entity_type, color in self.entity_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig("knowledge_graph/enhanced_graph_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plotly_graph(self, sample_size=500):
        """Create interactive Plotly visualization"""
        # Sample nodes for performance
        if len(self.graph.nodes()) > sample_size:
            nodes_to_include = list(self.graph.nodes())[:sample_size]
            subgraph = self.graph.subgraph(nodes_to_include)
        else:
            subgraph = self.graph
        
        # Calculate layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Prepare node data
        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        node_colors = [subgraph.nodes[node].get('color', '#DDA0DD') for node in subgraph.nodes()]
        node_sizes = [subgraph.degree(node) * 5 + 10 for node in subgraph.nodes()]
        node_text = [f"{node}<br>Type: {subgraph.nodes[node].get('type', 'Unknown')}<br>"
                    f"Category: {subgraph.nodes[node].get('category', 'Unknown')}<br>"
                    f"Connections: {subgraph.degree(node)}" 
                    for node in subgraph.nodes()]
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers',
                               hoverinfo='text',
                               text=node_text,
                               marker=dict(size=node_sizes,
                                         color=node_colors,
                                         line=dict(width=2, color='white')))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Dark Souls Knowledge Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        fig.write_html("knowledge_graph/interactive_graph.html")
        fig.show()
        
        print("💻 Interactive graph saved to knowledge_graph/interactive_graph.html")
    
    def create_category_subgraph(self, category, max_nodes=100):
        """Create visualization for specific category"""
        category_entities = self.entities_df[self.entities_df['category'] == category]
        
        if len(category_entities) == 0:
            print(f"❌ No entities found for category: {category}")
            return
        
        # Get nodes for this category
        category_nodes = category_entities['entity'].tolist()[:max_nodes]
        subgraph = self.graph.subgraph(category_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print(f"❌ No connected nodes found for category: {category}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Node properties
        node_colors = [subgraph.nodes[node].get('color', '#DDA0DD') for node in subgraph.nodes()]
        node_sizes = [subgraph.degree(node) * 200 + 100 for node in subgraph.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        plt.title(f"Dark Souls {category} Subgraph", size=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"knowledge_graph/{category.replace(' ', '_')}_subgraph.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 {category} subgraph saved to {filename}")
    
    def create_relationship_analysis(self):
        """Analyze and visualize relationship patterns"""
        # Relationship type distribution
        rel_types = self.relationships_df['type'].value_counts()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Relationship types pie chart
        ax1.pie(rel_types.values, labels=rel_types.index, autopct='%1.1f%%', 
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Relationship Types Distribution')
        
        # Entity types distribution
        entity_types = self.entities_df['type'].value_counts()
        bars = ax2.bar(entity_types.index, entity_types.values, 
                      color=[self.entity_colors.get(t, '#DDA0DD') for t in entity_types.index])
        ax2.set_title('Entity Types Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Category distribution
        categories = self.entities_df['category'].value_counts().head(10)
        ax3.barh(categories.index, categories.values, color='skyblue')
        ax3.set_title('Top 10 Categories')
        
        # Degree distribution
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        ax4.hist(degrees, bins=20, color='lightgreen', alpha=0.7)
        ax4.set_title('Node Degree Distribution')
        ax4.set_xlabel('Degree')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig("knowledge_graph/relationship_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def find_central_nodes(self, top_n=10):
        """Find and visualize most central nodes"""
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        
        # For large graphs, sample for other centrality measures
        if len(self.graph.nodes()) > 1000:
            sample_nodes = list(self.graph.nodes())[:1000]
            sample_graph = self.graph.subgraph(sample_nodes)
            betweenness = nx.betweenness_centrality(sample_graph, k=100)
            closeness = nx.closeness_centrality(sample_graph)
        else:
            betweenness = nx.betweenness_centrality(self.graph)
            closeness = nx.closeness_centrality(self.graph)
        
        # Get top nodes
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Degree centrality
        nodes, values = zip(*top_degree)
        ax1.barh(nodes, values, color='lightcoral')
        ax1.set_title('Top Nodes by Degree Centrality')
        ax1.set_xlabel('Degree Centrality')
        
        # Betweenness centrality
        nodes, values = zip(*top_betweenness)
        ax2.barh(nodes, values, color='lightblue')
        ax2.set_title('Top Nodes by Betweenness Centrality')
        ax2.set_xlabel('Betweenness Centrality')
        
        # Closeness centrality
        nodes, values = zip(*top_closeness)
        ax3.barh(nodes, values, color='lightgreen')
        ax3.set_title('Top Nodes by Closeness Centrality')
        ax3.set_xlabel('Closeness Centrality')
        
        plt.tight_layout()
        plt.savefig("knowledge_graph/centrality_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'degree': top_degree,
            'betweenness': top_betweenness,
            'closeness': top_closeness
        }
    
    def create_3d_visualization(self, sample_size=300):
        """Create 3D visualization using plotly"""
        # Sample for performance
        if len(self.graph.nodes()) > sample_size:
            nodes_to_include = list(self.graph.nodes())[:sample_size]
            subgraph = self.graph.subgraph(nodes_to_include)
        else:
            subgraph = self.graph
        
        # Get 3D positions using spring layout
        pos_2d = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Add random z-coordinate based on node degree
        node_x = [pos_2d[node][0] for node in subgraph.nodes()]
        node_y = [pos_2d[node][1] for node in subgraph.nodes()]
        node_z = [subgraph.degree(node) * 0.1 for node in subgraph.nodes()]
        
        node_colors = [subgraph.nodes[node].get('color', '#DDA0DD') for node in subgraph.nodes()]
        node_sizes = [subgraph.degree(node) * 3 + 5 for node in subgraph.nodes()]
        node_text = [f"{node}<br>Type: {subgraph.nodes[node].get('type', 'Unknown')}" 
                    for node in subgraph.nodes()]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title='3D Dark Souls Knowledge Graph',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Degree'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        fig.write_html("knowledge_graph/3d_graph.html")
        fig.show()
        print("🎯 3D visualization saved to knowledge_graph/3d_graph.html")
    
    def export_graph_data(self):
        """Export graph in various formats"""
        # GraphML format
        nx.write_graphml(self.graph, "knowledge_graph/darksouls_graph.graphml")
        
        # GML format
        nx.write_gml(self.graph, "knowledge_graph/darksouls_graph.gml")
        
        # Edge list
        nx.write_edgelist(self.graph, "knowledge_graph/darksouls_edgelist.txt")
        
        # Adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph)
        np.savetxt("knowledge_graph/adjacency_matrix.csv", adj_matrix.todense(), delimiter=",")
        
        print("📁 Graph exported in multiple formats:")
        print("  - GraphML: darksouls_graph.graphml")
        print("  - GML: darksouls_graph.gml") 
        print("  - Edge list: darksouls_edgelist.txt")
        print("  - Adjacency matrix: adjacency_matrix.csv")
    
    def analyze_communities(self):
        """Detect and visualize communities"""
        # Use the largest connected component for community detection
        largest_cc = max(nx.connected_components(self.graph), key=len)
        subgraph = self.graph.subgraph(largest_cc)
        
        if len(largest_cc) < 3:
            print("❌ Not enough connected nodes for community detection")
            return
        
        # Detect communities
        communities = nx.community.greedy_modularity_communities(subgraph)
        
        # Create color map for communities
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = {}
        
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = colors[i]
        
        # Visualize
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes by community
        for i, community in enumerate(communities):
            nx.draw_networkx_nodes(subgraph, pos, 
                                 nodelist=list(community),
                                 node_color=[colors[i]], 
                                 node_size=100,
                                 alpha=0.8,
                                 label=f'Community {i+1} ({len(community)} nodes)')
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
        
        plt.title(f'Community Detection - {len(communities)} Communities Found')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("knowledge_graph/community_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🏘️ Found {len(communities)} communities")
        for i, community in enumerate(communities):
            print(f"  Community {i+1}: {len(community)} nodes")
            if len(community) <= 10:  # Show members for small communities
                print(f"    Members: {', '.join(list(community))}")

def main():
    """Main function to demonstrate the graph builder"""
    print("🎮 Dark Souls Graph Builder")
    print("=" * 50)
    
    # Initialize graph builder
    builder = DarkSoulsGraphBuilder()
    
    # Get statistics
    stats = builder.get_graph_statistics()
    print("\n📊 Graph Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Basic graph
    builder.create_basic_visualization(layout='spring')
    
    # Interactive graph
    builder.create_interactive_plotly_graph(sample_size=300)
    
    # Relationship analysis
    builder.create_relationship_analysis()
    
    # Central nodes analysis
    central_nodes = builder.find_central_nodes()
    
    # Community analysis
    builder.analyze_communities()
    
    # 3D visualization
    builder.create_3d_visualization()
    
    # Create category subgraphs for major categories
    major_categories = ['Weapons', 'Bosses', 'NPCs', 'Consumables']
    for category in major_categories:
        if category in builder.entities_df['category'].values:
            builder.create_category_subgraph(category)
    
    # Export graph data
    builder.export_graph_data()
    
    print("\n✅ Graph building complete!")
    print("📁 Check the knowledge_graph/ directory for all visualizations and exports")

if __name__ == "__main__":
    main()
