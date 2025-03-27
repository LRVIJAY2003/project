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





"""
Remedy API connection test script.
Run this to verify Remedy integration is working correctly.
"""
import os
import sys
import json
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

def test_remedy_connection():
    """Test Remedy API connectivity and access."""
    
    print("=" * 50)
    print("REMEDY CONNECTION TEST")
    print("=" * 50)
    
    # Check environment variables
    remedy_server = os.getenv("REMEDY_SERVER")
    remedy_api_base = os.getenv("REMEDY_API_BASE")
    remedy_username = os.getenv("REMEDY_USERNAME")
    remedy_password = os.getenv("REMEDY_PASSWORD")
    
    # Validate configuration
    if not remedy_server:
        print("❌ REMEDY_SERVER not set in environment")
        return False
    
    if not remedy_api_base:
        print("❌ REMEDY_API_BASE not set in environment")
        return False
    
    if not remedy_username:
        print("❌ REMEDY_USERNAME not set in environment")
        return False
    
    if not remedy_password:
        print("❌ REMEDY_PASSWORD not set in environment")
        return False
    
    print(f"✅ Configuration found:")
    print(f"   Server: {remedy_server}")
    print(f"   API Base: {remedy_api_base}")
    print(f"   Username: {remedy_username}")
    print(f"   Password: {'*' * 8 if remedy_password else 'None'}")
    
    # Try to import and initialize Remedy client
    try:
        from modules.data_sources.remedy import RemedyClient
        
        print("\nInitializing Remedy client...")
        client = RemedyClient(
            base_url=remedy_api_base,
            username=remedy_username,
            password=remedy_password
        )
        print("✅ Remedy client initialized")
        
        # Try to get authentication token
        print("\nGetting authentication token...")
        token = client.get_token()
        
        if token:
            print(f"✅ Successfully obtained authentication token")
            print(f"   Token: {token[:10]}...{token[-10:] if len(token) > 20 else ''}")
        else:
            print("❌ Failed to obtain authentication token")
            return False
        
        # Try to search for incidents
        print("\nSearching for incidents...")
        incidents = client.search_incidents(limit=5)
        
        if incidents:
            print(f"✅ Successfully retrieved {len(incidents)} incidents")
            print("\nSample incidents:")
            for incident in incidents:
                incident_id = incident.get('id', 'Unknown')
                incident_values = incident.get('values', {})
                summary = incident_values.get('Summary', 'No summary available')
                print(f"   - {incident_id}: {summary[:50]}...")
            
            # Try to get details for first incident
            if incidents:
                first_incident = incidents[0]
                incident_id = first_incident.get('id')
                print(f"\nFetching details for incident {incident_id}...")
                incident_details = client.get_incident(incident_id)
                
                if incident_details:
                    print(f"✅ Successfully retrieved incident details")
                else:
                    print("⚠️ Failed to retrieve incident details")
        else:
            print("⚠️ No incidents found or access denied")
        
        # Try to search for change requests
        print("\nSearching for change requests...")
        changes = client.search_change_requests(limit=5)
        
        if changes:
            print(f"✅ Successfully retrieved {len(changes)} change requests")
        else:
            print("⚠️ No change requests found or access denied")
        
        # Try to search for knowledge articles
        print("\nSearching for knowledge articles...")
        articles = client.search_knowledge_articles(limit=5)
        
        if articles:
            print(f"✅ Successfully retrieved {len(articles)} knowledge articles")
        else:
            print("⚠️ No knowledge articles found or access denied")
        
        print("\n✅ Remedy connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Remedy test: {e}")
        print("\nDetailed error information:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_remedy_connection()
    if not success:
        print("\n⚠️ Remedy connection test failed. Please check your configuration and connectivity.")
        sys.exit(1)
    else:
        print("\n✅ All Remedy connection tests passed!")