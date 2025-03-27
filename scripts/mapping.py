import openai
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# File paths (matching your structure)
METADATA_FILE = '../data/metadata.csv'
MAPPING_RULES_FILE = '../data/mapping_rules.csv'
OUTPUT_FILE = '../data/ai_generated_mapping1.csv'

def load_data():
    """Load input files with error handling"""
    try:
        metadata = pd.read_csv(METADATA_FILE,delimiter='|')
        mapping_rules = pd.read_csv(MAPPING_RULES_FILE, delimiter='|')
        return metadata, mapping_rules
    except Exception as e:
        print(f"Error loading files: {e}")
        raise

def generate_sql_logic(row, metadata):
    """Generate SQL implementation from business rules using AI"""
    source_meta = metadata[
        (metadata['source_table'] == row['source_table']) & 
        (metadata['source_field'] == row['source_field'])
    ].iloc[0]
    
    prompt = f"""
    Create SQL transformation logic for this data mapping:
    
    Source Table: {row['source_table']}
    Source Field: {row['source_field']} ({source_meta['data_type']})
    Description: {source_meta['description']}
    Target Field: {row['target_field']}
    Business Rule: {row['business_rule']}
    
    Requirements:
    1. Use standard SQL (ANSI)
    2. Handle NULLs appropriately
    3. Include data type conversion if needed
    4. Assume tables are joined via: a.loan_id = c.cap_product_id
    5. Return ONLY executable SQL
    
    Example Output: COALESCE(a.loan_id, c.cap_product_id)
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating SQL for {row['source_field']}: {e}")
        return None

def process_mappings(metadata, mapping_rules):
    """Generate complete mapping table"""
    results = []
    
    # Create base mapping table
    for _, rule in mapping_rules.iterrows():
        # ACBS mapping
        if not pd.isna(rule['acbs_field']):
            results.append({
                'source_table': 'acbs_loan',
                'source_field': rule['acbs_field'],
                'target_field': rule['target_field'],
                'business_rule': rule['transformation_logic']
            })
        
        # CAP mapping
        if not pd.isna(rule['cap_field']):
            results.append({
                'source_table': 'cap_product_info',
                'source_field': rule['cap_field'],
                'target_field': rule['target_field'],
                'business_rule': rule['transformation_logic']
            })
    
    # Enhance with SQL logic
    mapping_df = pd.DataFrame(results)
    mapping_df['sql_logic'] = mapping_df.apply(
        lambda x: generate_sql_logic(x, metadata), 
        axis=1
    )
    
    return mapping_df[['source_table', 'source_field', 'target_field', 'business_rule', 'sql_logic']]

def main():
    print("Starting mapping generation...")
    metadata, mapping_rules = load_data()
    final_mapping = process_mappings(metadata, mapping_rules)
    
    # Save results
    final_mapping.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated mapping to {OUTPUT_FILE}")
    print("\nSample output:")
    print(final_mapping.head().to_markdown())

if __name__ == "__main__":
    main()