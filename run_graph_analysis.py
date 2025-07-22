
#!/usr/bin/env python3
"""
Dark Souls Knowledge Graph Analysis Runner
=========================================

This script demonstrates various graph analysis and visualization capabilities.
"""

import os
import sys
from graph_builder import DarkSoulsGraphBuilder

def main():
    print("🎮 Dark Souls Knowledge Graph Analysis")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists("knowledge_graph/entities.csv"):
        print("❌ Knowledge graph data not found!")
        print("Please run main.py first to generate the data.")
        return
    
    print("1. 📊 Full Graph Analysis")
    print("2. 🎯 Category-specific Analysis")
    print("3. 🌐 Interactive Visualizations")
    print("4. 📈 Statistical Analysis")
    print("5. 🏘️ Community Detection")
    print("6. 🚀 Run Streamlit Dashboard")
    print("7. ⚡ Quick Demo (All visualizations)")
    print()
    
    choice = input("Select option (1-7): ").strip()
    
    # Initialize graph builder
    builder = DarkSoulsGraphBuilder()
    
    if choice == "1":
        print("\n📊 Creating full graph analysis...")
        builder.create_basic_visualization()
        stats = builder.get_graph_statistics()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    elif choice == "2":
        print("\n🎯 Available categories:")
        categories = builder.entities_df['category'].unique()
        for i, cat in enumerate(sorted(categories), 1):
            count = len(builder.entities_df[builder.entities_df['category'] == cat])
            print(f"  {i}. {cat} ({count} entities)")
        
        try:
            cat_choice = int(input("\nSelect category number: ")) - 1
            selected_category = sorted(categories)[cat_choice]
            builder.create_category_subgraph(selected_category)
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == "3":
        print("\n🌐 Creating interactive visualizations...")
        builder.create_interactive_plotly_graph()
        builder.create_3d_visualization()
        print("✅ Interactive visualizations created!")
        print("📁 Check knowledge_graph/interactive_graph.html and knowledge_graph/3d_graph.html")
    
    elif choice == "4":
        print("\n📈 Running statistical analysis...")
        builder.create_relationship_analysis()
        central_nodes = builder.find_central_nodes()
        print("\n🔝 Most central entities:")
        print("By Degree Centrality:")
        for node, score in central_nodes['degree'][:5]:
            print(f"  {node}: {score:.4f}")
    
    elif choice == "5":
        print("\n🏘️ Detecting communities...")
        builder.analyze_communities()
    
    elif choice == "6":
        print("\n🚀 Starting Streamlit dashboard...")
        print("Dashboard will open in your browser...")
        os.system("streamlit run graph_dashboard.py")
    
    elif choice == "7":
        print("\n⚡ Running complete analysis...")
        
        # Basic visualizations
        print("Creating basic graph visualization...")
        builder.create_basic_visualization()
        
        # Interactive graphs
        print("Creating interactive visualizations...")
        builder.create_interactive_plotly_graph(sample_size=200)
        
        # Statistical analysis
        print("Running statistical analysis...")
        builder.create_relationship_analysis()
        
        # Centrality analysis
        print("Analyzing central nodes...")
        builder.find_central_nodes()
        
        # Community detection
        print("Detecting communities...")
        builder.analyze_communities()
        
        # Export data
        print("Exporting graph data...")
        builder.export_graph_data()
        
        # Create category subgraphs for major categories
        major_categories = ['Weapons', 'Bosses', 'NPCs', 'Consumables']
        for category in major_categories:
            if category in builder.entities_df['category'].values:
                print(f"Creating {category} subgraph...")
                builder.create_category_subgraph(category)
        
        print("\n✅ Complete analysis finished!")
        print("📁 All visualizations saved to knowledge_graph/ directory")
    
    else:
        print("❌ Invalid option selected")

if __name__ == "__main__":
    main()
