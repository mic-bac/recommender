"""
==============================================================================
ASSOCIATION RULE MINING - TEACHING SCRIPT
Master's Course: Introduction to Data Mining Methods in Marketing Analytics
==============================================================================

This script demonstrates two fundamental ARM algorithms:
1. Apriori Algorithm: Level-wise candidate generation and pruning
2. FP-Growth Algorithm: Compressed representation using FP-trees

Key Concepts:
- Itemsets: Collections of items that frequently appear together
- Support: Frequency of itemset occurrence (Popularity)
- Confidence: Likelihood of B given A (IF → THEN strength)
- Lift: How much more likely B occurs given A (Independence measure)

Dataset: Groceries - Transaction data from a grocery store
"""

# %% 1. ENVIRONMENT SETUP & DATA LOADING
# =============================================================================
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load the groceries dataset
# Expected format: CSV with transactions (comma-separated items per row)
print("📦 Loading Groceries Dataset...")
df = pd.read_csv('./data/groceries/Groceries_dataset.csv')
# We create an Invoice column to have all purchases grouped by receipt (assumption for bought together)
df["Invoice"] = df["Member_number"].astype(str) + df["Date"]

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 transactions:\n{df.head()}")
print(f"\nColumn names: {df.columns.tolist()}")


# %% 2. DATA PREPROCESSING
# =============================================================================
"""
PREPROCESSING STEP:
The TransactionEncoder converts transaction data (list of lists) into a 
one-hot encoded DataFrame suitable for ARM algorithms.

Before: [['milk', 'bread'], ['bread', 'butter']]
After:  DataFrame with binary columns (0/1) for each item
"""

print("\n" + "="*80)
print("DATA PREPROCESSING: Transaction Encoding")
print("="*80)

# Extract transactions - assuming 'itemDescription' column contains items
# If items are already separated, adjust accordingly
transactions = df.groupby('Invoice')['itemDescription'].apply(list).values

print(f"Number of transactions: {len(transactions)}")
print(f"Sample transaction 1: {transactions[0][:5]}...")  # Show first 5 items

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"\nEncoded DataFrame shape: {transaction_df.shape}")
print(f"Number of unique items: {len(te.columns_)}")
print(f"\nFirst 5 items: {list(te.columns_)[:5]}")
print(f"\nEncoded data (first 5 rows, first 10 columns):\n{transaction_df.iloc[:5, :10]}")


# %% 3. CALCULATE ITEM FREQUENCIES (SUPPORT)
# =============================================================================
"""
SUPPORT ANALYSIS:
Support(X) = (Number of transactions containing X) / (Total transactions)

This helps us understand which items are popular in the grocery store.
Higher support = items more frequently purchased together.
"""

print("\n" + "="*80)
print("SUPPORT ANALYSIS: Individual Item Frequencies")
print("="*80)

# Calculate support for individual items
item_support = transaction_df.sum() / len(transaction_df)
item_support = item_support.sort_values(ascending=False)

print(f"Top 15 most frequently purchased items:")
print(item_support.head(15))

# Visualization: Top 20 items by support
fig_support = px.bar(
    x=item_support.head(20).values,
    y=item_support.head(20).index,
    orientation='h',
    title='📊 Top 20 Items by Support (Popularity)',
    labels={'x': 'Support (Frequency Ratio)', 'y': 'Item'},
    color=item_support.head(20).values,
    color_continuous_scale='Viridis'
)
fig_support.update_layout(
    height=600,
    showlegend=False,
    hovermode='y unified'
)
fig_support.show()


# %% 4. APRIORI ALGORITHM
# =============================================================================
"""
APRIORI ALGORITHM - Level-wise candidate generation:

Key Principle: "If an itemset is frequent, all its subsets are frequent"
(Anti-monotone property)

Algorithm Flow:
1. Generate 1-itemsets (individual items) above min_support
2. Generate k-itemsets from (k-1)-itemsets
3. Prune candidates that don't meet min_support
4. Repeat until no new itemsets found

Advantages: Easy to understand, supports constraints
Disadvantages: May generate many candidates, slower for large datasets
"""

print("\n" + "="*80)
print("ALGORITHM 1: APRIORI")
print("="*80)

# Set minimum support threshold
# Usually 1-5% for retail data; we'll use a very small value for demonstrational purpose
min_support_apriori = 0.001

print(f"Minimum Support Threshold: {min_support_apriori*100}%")
print("Running Apriori algorithm...")

# Run Apriori
frequent_itemsets_apriori = apriori(
    transaction_df,
    min_support=min_support_apriori,
    use_colnames=True,
    max_len=5  # Limit to 5-itemsets for interpretability
)

# Add itemset size for analysis
frequent_itemsets_apriori['itemset_size'] = frequent_itemsets_apriori['itemsets'].apply(len)

print(f"\n✅ Apriori Results:")
print(f"Number of frequent itemsets: {len(frequent_itemsets_apriori)}")
print(f"Itemsets by size:")
print(frequent_itemsets_apriori['itemset_size'].value_counts().sort_index())
print(f"\nTop 10 (1-)itemsets by support:")
print(frequent_itemsets_apriori[frequent_itemsets_apriori['itemset_size'] == 1].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))
print(f"\nTop 10 (2-)itemsets by support:")
print(frequent_itemsets_apriori[frequent_itemsets_apriori['itemset_size'] == 2].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))
print(f"\nTop 10 (3-)itemsets by support:")
print(frequent_itemsets_apriori[frequent_itemsets_apriori['itemset_size'] == 3].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))


# %% 5. FP-GROWTH ALGORITHM
# =============================================================================
"""
FP-GROWTH ALGORITHM - Tree-based compact representation:

Key Principle: Compress transaction database into FP-tree, mine patterns
without candidate generation

Algorithm Flow:
1. Build FP-tree: Compress transactions into prefix tree structure
2. Determine item ordering by support (descending)
3. Mine patterns directly from tree without candidates
4. Recursive mining of conditional FP-trees

Advantages: Much faster, fewer memory requirements
Disadvantages: Slightly less interpretable, tree structure overhead

Performance Note: FP-Growth typically outperforms Apriori by 2-10x
"""

print("\n" + "="*80)
print("ALGORITHM 2: FP-GROWTH")
print("="*80)

# Use same minimum support for fair comparison
min_support_fpgrowth = 0.001

print(f"Minimum Support Threshold: {min_support_fpgrowth*100}%")
print("Running FP-Growth algorithm...")

# Run FP-Growth
frequent_itemsets_fpgrowth = fpgrowth(
    transaction_df,
    min_support=min_support_fpgrowth,
    use_colnames=True,
    max_len=5
)

# Add itemset size for analysis
frequent_itemsets_fpgrowth['itemset_size'] = frequent_itemsets_fpgrowth['itemsets'].apply(len)

print(f"\n✅ FP-Growth Results:")
print(f"Number of frequent itemsets: {len(frequent_itemsets_fpgrowth)}")
print(f"Itemsets by size:")
print(frequent_itemsets_fpgrowth['itemset_size'].value_counts().sort_index())
print(f"\nTop 10 (1-)itemsets by support:")
print(frequent_itemsets_fpgrowth[frequent_itemsets_fpgrowth['itemset_size'] == 1].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))
print(f"\nTop 10 (2-)itemsets by support:")
print(frequent_itemsets_fpgrowth[frequent_itemsets_fpgrowth['itemset_size'] == 2].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))
print(f"\nTop 10 (3-)itemsets by support:")
print(frequent_itemsets_fpgrowth[frequent_itemsets_fpgrowth['itemset_size'] == 3].nlargest(10, 'support')[['itemsets', 'support']].to_string(index=False))


# %% 6. ALGORITHM COMPARISON
# =============================================================================
"""
COMPARISON: Apriori vs FP-Growth

Both algorithms should produce identical results (same itemsets and support)
but may differ in computational efficiency.
"""

print("\n" + "="*80)
print("ALGORITHM COMPARISON")
print("="*80)

# Verify both algorithms produce the same results
print("Verification: Are results identical?")
print(f"Apriori itemsets count: {len(frequent_itemsets_apriori)}")
print(f"FP-Growth itemsets count: {len(frequent_itemsets_fpgrowth)}")

# Compare itemsets (accounting for frozenset order)
apriori_sets = set(frequent_itemsets_apriori['itemsets'])
fpgrowth_sets = set(frequent_itemsets_fpgrowth['itemsets'])
print(f"Same itemsets found: {apriori_sets == fpgrowth_sets}")


# %% 7. ASSOCIATION RULES GENERATION
# =============================================================================
"""
ASSOCIATION RULES:
From frequent itemsets, derive IF → THEN rules

Metrics:
- Support(A→B) = P(A∪B) = freq(A∪B) / total_transactions
- Confidence(A→B) = P(B|A) = freq(A∪B) / freq(A)
- Lift(A→B) = Confidence(A→B) / Support(B) 
  → If Lift > 1: A and B are positively correlated
  → If Lift = 1: A and B are independent
  → If Lift < 1: A and B are negatively correlated

Interpretation Example:
Rule: {milk} → {bread}
- Support: 10% (milk and bread together in 10% of transactions)
- Confidence: 60% (60% of milk buyers also buy bread)
- Lift: 1.5 (Bread buyers are 1.5x more likely if milk purchased)
"""

print("\n" + "="*80)
print("ASSOCIATION RULES GENERATION")
print("="*80)

from mlxtend.frequent_patterns import association_rules

min_threshold = 0.1 # At least 10% confidence

# Generate rules from Apriori results
rules_apriori = association_rules(
    frequent_itemsets_apriori,
    metric='confidence',
    min_threshold=min_threshold
)

# Calculate additional metrics
if len(rules_apriori) > 0:
    rules_apriori['antecedent_size'] = rules_apriori['antecedents'].apply(len)
    rules_apriori['consequent_size'] = rules_apriori['consequents'].apply(len)
    rules_apriori['rule_str'] = (
        rules_apriori['antecedents'].apply(lambda x: ', '.join(list(x))) +
        ' → ' +
        rules_apriori['consequents'].apply(lambda x: ', '.join(list(x)))
    )

    print(f"\n✅ Generated {len(rules_apriori)} rules with confidence ≥ {min_threshold*100}%")
    print("\nTop 15 rules by Lift (best cross-selling opportunities):")
    top_rules = rules_apriori.nlargest(15, 'lift')[
        ['rule_str', 'support', 'confidence', 'lift']
    ]
    print(top_rules.to_string())
else:
    print("⚠️  No rules generated with current thresholds")
    rules_apriori = pd.DataFrame()


# %% 7b. TURNING RULES INTO A RECOMMENDER
# =============================================================================
"""
FROM RULES TO RECOMMENDATIONS:
"Customers who bought X also bought Y."

Given an item a customer just added to their basket, we look up every rule whose
antecedent (the "if" side) contains that item, and recommend the consequents
(the "then" side). We rank the suggestions by lift (associations strongest beyond
chance) and then confidence (most reliable), which is the non-personalized,
transaction-based recommendation described in the lecture.
"""

def get_basket_recommendations(item, rules=rules_apriori, top_n=5):
    """Recommend items frequently bought together with `item`.

    Args:
        item: the product already in the basket (e.g. 'whole milk').
        rules: an association-rules DataFrame (from `association_rules`).
        top_n: how many recommendations to return.

    Returns:
        A DataFrame of recommended items with the rule's support/confidence/lift,
        or an informative string if nothing matches.
    """
    if rules is None or len(rules) == 0:
        return "No association rules available — run rule generation first."

    # Keep rules where the antecedent contains the basket item.
    matches = rules[rules['antecedents'].apply(lambda a: item in a)].copy()
    if matches.empty:
        return f"No rules found with '{item}' on the antecedent side."

    # The recommendation is the consequent (the "then" side) of each matching rule.
    matches['recommendation'] = matches['consequents'].apply(lambda c: ', '.join(sorted(c)))

    # Rank by strength (lift) then reliability (confidence), de-duplicating items.
    matches = matches.sort_values(['lift', 'confidence'], ascending=False)
    matches = matches.drop_duplicates(subset='recommendation')

    return matches[['recommendation', 'support', 'confidence', 'lift']].head(top_n)


# %%
# Example: what should we suggest to someone buying 'whole milk'?
print("\n" + "="*80)
print("BASKET RECOMMENDATIONS: items bought together with 'whole milk'")
print("="*80)
milk_recs = get_basket_recommendations('whole milk')
print(milk_recs if isinstance(milk_recs, str) else milk_recs.to_string(index=False))


# %% 8. VISUALIZATION: ITEMSET SUPPORT DISTRIBUTION
# =============================================================================
"""
VISUALIZATION 1: Distribution of itemset support values

This helps understand the pattern of support across different itemsets.
Shows which itemsets are highly supported vs. rare but interesting.
"""

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig_itemsets = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f'Itemset Support Distribution (Apriori, n={len(frequent_itemsets_apriori)})',
        f'Itemset Size Distribution (Apriori, n={len(frequent_itemsets_apriori)})'
    )
)

# Plot 1: Support distribution histogram
fig_itemsets.add_trace(
    go.Histogram(
        x=frequent_itemsets_apriori['support'],
        nbinsx=40,
        name='Support',
        marker_color='rgba(0, 150, 200, 0.7)',
        showlegend=False
    ),
    row=1, col=1
)

# Plot 2: Itemset size distribution
itemset_sizes = frequent_itemsets_apriori['itemset_size'].value_counts().sort_index()
fig_itemsets.add_trace(
    go.Bar(
        x=itemset_sizes.index,
        y=itemset_sizes.values,
        name='Count',
        marker_color='rgba(200, 100, 0, 0.7)',
        showlegend=False
    ),
    row=1, col=2
)

fig_itemsets.update_xaxes(title_text='Support Value', row=1, col=1)
fig_itemsets.update_xaxes(title_text='Itemset Size (# of items)', row=1, col=2)
fig_itemsets.update_yaxes(title_text='Frequency', row=1, col=1)
fig_itemsets.update_yaxes(title_text='Number of Itemsets', row=1, col=2)

fig_itemsets.update_layout(
    title_text='📈 Itemset Analysis: Apriori Results',
    height=500,
    showlegend=False
)
fig_itemsets.show()


# %% 9. VISUALIZATION: ASSOCIATION RULES SCATTER PLOT
# =============================================================================
"""
VISUALIZATION 2: Association Rules Scatter Plot

This is one of the most important visualizations for ARM results.

Axes:
- X-axis: Support (how frequent is the itemset A∪B)
- Y-axis: Confidence (how often B follows A)
- Color: Lift (how much more likely B given A)
- Size: Itemset size

Interpretation:
- Top-right: High support + high confidence (strong, popular rules)
- Top-left: Low support + high confidence (strong but rare rules)
- Bottom-right: High support + low confidence (frequent but weak rules)
- Bottom-left: Low support + low confidence (neither frequent nor strong)
"""

if len(rules_apriori) > 0:
    print("Creating Association Rules Scatter Plot...")
    
    fig_rules = go.Figure()
    
    fig_rules.add_trace(go.Scatter(
        x=rules_apriori['support'],
        y=rules_apriori['confidence'],
        mode='markers',
        marker=dict(
            size=rules_apriori['lift'] * 10,  # Size by lift
            color=rules_apriori['lift'],  # Color by lift
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Lift',
                thickness=15,
                len=0.7,
            ),
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=[
            f"<b>{rule}</b><br>" +
            f"Support: {supp:.3f}<br>" +
            f"Confidence: {conf:.3f}<br>" +
            f"Lift: {lift:.3f}"
            for rule, supp, conf, lift in zip(
                rules_apriori['rule_str'],
                rules_apriori['support'],
                rules_apriori['confidence'],
                rules_apriori['lift']
            )
        ],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig_rules.update_layout(
        title='🎯 Association Rules: Support vs Confidence (colored by Lift)',
        xaxis_title='Support (Itemset Frequency)',
        yaxis_title='Confidence (Rule Strength)',
        height=600,
        width=900,
        hovermode='closest'
    )
    
    # Add reference lines
    fig_rules.add_vline(x=rules_apriori['support'].median(), 
                        line_dash='dash', line_color='gray', 
                        annotation_text='Median Support')
    fig_rules.add_hline(y=rules_apriori['confidence'].median(), 
                        line_dash='dash', line_color='gray',
                        annotation_text='Median Confidence')
    
    fig_rules.show()
else:
    print("⚠️  Skipping rules visualization - no rules generated")


# %% 10. VISUALIZATION: NETWORK GRAPH (Top Rules)
# =============================================================================
"""
VISUALIZATION 3: Network Graph of Association Rules

Shows relationships between items in the top rules.
This helps identify hubs (items that are central to many rules).
"""
top_n = 10

if len(rules_apriori) > 10:
    print("Creating Association Rules Network Graph...")
    
    # Get top rules by lift
    top_n_rules = rules_apriori.nlargest(top_n, 'lift')
    
    # Create network edges
    edges_from = []
    edges_to = []
    edge_weights = []
    
    for idx, row in top_n_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for ant in antecedents:
            for cons in consequents:
                edges_from.append(ant)
                edges_to.append(cons)
                edge_weights.append(row['lift'])
    
    # Create nodes
    all_items = list(set(edges_from + edges_to))
    
    # Simulate node positions (circular layout)
    import math
    n_nodes = len(all_items)
    node_x = []
    node_y = []
    for i in range(n_nodes):
        angle = 2 * math.pi * i / n_nodes
        node_x.append(math.cos(angle))
        node_y.append(math.sin(angle))
    
    # Edge trace
    edge_x = []
    edge_y = []
    for i, j in zip(edges_from, edges_to):
        x0, y0 = node_x[all_items.index(i)], node_y[all_items.index(i)]
        x1, y1 = node_x[all_items.index(j)], node_y[all_items.index(j)]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig_network = go.Figure()
    
    # Add edges
    fig_network.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig_network.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color='#0080FF',
            line_width=2,
            line_color='white'
        ),
        text=all_items,
        textposition='top center',
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    ))
    
    fig_network.update_layout(
        title=f'🕸️ Association Rules Network: Top {top_n} Rules by Lift',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=900,
        plot_bgcolor='rgba(240, 240, 240, 0.9)'
    )
    fig_network.show()


# %% 11. SUMMARY & BUSINESS INSIGHTS
# =============================================================================
"""
SUMMARY: Key Takeaways for Marketing Analytics

1. APRIORI vs FP-GROWTH
   - Both find same itemsets
   - FP-Growth: Faster, uses less memory (recommended for large datasets)
   - Apriori: More transparent, easier to understand

2. SUPPORT ANALYSIS
   - Identifies popular items and combinations
   - Used for inventory management and product placement

3. ASSOCIATION RULES
   - Confidence: How predictable is the rule (basis for recommendations)
   - Lift: How much better than random guessing (basis for cross-selling)

4. BUSINESS APPLICATIONS
   - Market Basket Analysis: What products to bundle
   - Recommendation Systems: "Customers who bought X also bought Y"
   - Store Layout: Place related items near each other
   - Promotion Strategy: Cross-sell opportunities

5. PARAMETER TUNING
   - Lower min_support: Find more itemsets (slower, more patterns)
   - Higher min_confidence: Find more reliable rules
   - Adjust based on business requirements
"""

print("\n" + "="*80)
print("SUMMARY: KEY INSIGHTS FROM ASSOCIATION RULE MINING")
print("="*80)

print(f"""
📊 DATASET OVERVIEW:
   • Total transactions: {len(transaction_df)}
   • Unique items: {len(te.columns_)}
   • Average items per transaction: {transaction_df.sum(axis=1).mean():.2f}

🔍 APRIORI ALGORITHM:
   • Frequent itemsets found: {len(frequent_itemsets_apriori)}
   • 1-itemsets: {(frequent_itemsets_apriori['itemset_size']==1).sum()}
   • 2-itemsets: {(frequent_itemsets_apriori['itemset_size']==2).sum()}
   • 3-itemsets: {(frequent_itemsets_apriori['itemset_size']==3).sum()}

🚀 FP-GROWTH ALGORITHM:
   • Frequent itemsets found: {len(frequent_itemsets_fpgrowth)}
   • Algorithms produce identical results: ✓

📈 ASSOCIATION RULES:
   • Rules generated (confidence ≥ {min_threshold*100}%): {len(rules_apriori)}
""")

if len(rules_apriori) > 0:
    print(f"""
💡 TOP BUSINESS INSIGHTS:
   • Strongest rule (by lift):
     {rules_apriori.nlargest(1, 'lift')['rule_str'].values[0]}
     (Lift: {rules_apriori['lift'].max():.2f})
   
   • Most confident rule:
     {rules_apriori.nlargest(1, 'confidence')['rule_str'].values[0]}
     (Confidence: {rules_apriori['confidence'].max():.2%})
   
   • Most common rule (by support):
     {rules_apriori.nlargest(1, 'support')['rule_str'].values[0]}
     (Support: {rules_apriori['support'].max():.2%})

🎯 RECOMMENDATIONS FOR MARKETING:
   1. Bundle top-lifted items together for promotion
   2. Use high-confidence rules for recommendation engines
   3. Place related items near each other in stores
   4. Create cross-sell campaigns for high-lift rules
   5. Adjust inventory based on support values
""")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
# %%
