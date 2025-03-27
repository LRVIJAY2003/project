"""
Confluence API connection test script.
Run this to verify Confluence integration is working correctly.
"""
import os
import sys
import json
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

def test_confluence_connection():
    """Test Confluence API connectivity and access."""
    
    print("=" * 50)
    print("CONFLUENCE CONNECTION TEST")
    print("=" * 50)
    
    # Check environment variables
    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_user = os.getenv("CONFLUENCE_USER_ID")
    confluence_token = os.getenv("CONFLUENCE_API_TOKEN")
    confluence_space = os.getenv("CONFLUENCE_SPACE_ID")
    
    # Validate configuration
    if not confluence_url:
        print("❌ CONFLUENCE_URL not set in environment")
        return False
    
    if not confluence_user:
        print("❌ CONFLUENCE_USER_ID not set in environment")
        return False
    
    if not confluence_token:
        print("❌ CONFLUENCE_API_TOKEN not set in environment")
        return False
    
    print(f"✅ Configuration found:")
    print(f"   URL: {confluence_url}")
    print(f"   User: {confluence_user}")
    print(f"   Token: {'*' * 8 + confluence_token[-4:] if confluence_token else 'None'}")
    print(f"   Space: {confluence_space or 'Not specified'}")
    
    # Try to import and initialize Confluence client
    try:
        from modules.data_sources.confluence import ConfluenceClient
        
        print("\nInitializing Confluence client...")
        client = ConfluenceClient(
            base_url=confluence_url,
            user_id=confluence_user,
            api_token=confluence_token
        )
        print("✅ Confluence client initialized")
        
        # Try to get spaces
        print("\nFetching spaces...")
        spaces = client.get_spaces(limit=5)
        
        if spaces:
            print(f"✅ Successfully retrieved {len(spaces)} spaces")
            print("\nAvailable spaces:")
            for space in spaces:
                print(f"   - {space.get('key', 'Unknown')}: {space.get('name', 'Unnamed')}")
        else:
            print("⚠️ No spaces found or access denied")
        
        # Try to get pages if a space is specified
        if confluence_space:
            print(f"\nFetching pages from space {confluence_space}...")
            pages = client.get_pages(space_key=confluence_space, limit=5)
            
            if pages:
                print(f"✅ Successfully retrieved {len(pages)} pages")
                print("\nSample pages:")
                for page in pages:
                    print(f"   - {page.get('id', 'Unknown')}: {page.get('title', 'Untitled')}")
                
                # Try to get content from first page
                if pages:
                    first_page = pages[0]
                    page_id = first_page.get('id')
                    print(f"\nFetching content from page {page_id}...")
                    content = client.get_page_content(page_id)
                    
                    if content:
                        excerpt = content[:100] + "..." if len(content) > 100 else content
                        print(f"✅ Successfully retrieved content: {excerpt}")
                    else:
                        print("⚠️ No content found for page")
            else:
                print("⚠️ No pages found in space or access denied")
        
        print("\n✅ Confluence connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Confluence test: {e}")
        print("\nDetailed error information:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_confluence_connection()
    if not success:
        print("\n⚠️ Confluence connection test failed. Please check your configuration and connectivity.")
        sys.exit(1)
    else:
        print("\n✅ All Confluence connection tests passed!")