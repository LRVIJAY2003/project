You're right! Let's debug the backend. The issue is likely in how we're extracting data from the XML in the PostgreSQL query or how we're processing the results. Let me provide a comprehensive fix for `app.py`.

## Updated app.py (Backend Fix - Complete Replacement)

```python
from flask import Flask, render_template, request, jsonify
from google.cloud.sql.connector import Connector
import google.auth
from sqlalchemy import create_engine, text
import oracledb
import sqlalchemy
import re
import os
from ldap3 import Server, Connection, ALL
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Database configurations
CA_CERT_PATH = "./new.pem"  # Update this path for your environment

DB_INSTANCES = {
    'oscar': {
        'project': 'pri-dv-oscar-0302',
        'region': 'us-central1',
        'instance': 'csal-dv-usc1-1316-oscar-0004-m',
        'database': 'padk',
        'user': 'lakshya.vijay@cmegroup.com',
        'schema': 'dv01cosrs',
        'description': 'OSCAR Database Instance'
    }
}

ORACLE_INSTANCES = {
    'copper': {
        'user': 'QF44CCRDO',
        'password': 'Felty_22_PAMPA_TRJ_44',
        'url': 'ldap://ORAQALDAP:3060/DCCRDC,cn=OracleContext,dc=world',
        'description': 'CoPPER Database'
    },
    'edb': {
        'user': 'DV02CEDBO',
        'password': 'eh02eedbo',
        'url': 'ldap://ORAQALDAP:3060/DCEDBA,cn=OracleContext,dc=world',
        'description': 'EDB Database'
    },
    'star': {
        'user': 'DV99CFASO',
        'password': 'DV99CFASOJ',
        'url': 'ldap://ORAQALDAP:3060/DCFASA,cn=OracleContext,dc=world',
        'description': 'STAR Database'
    }
}


class DatabaseConnector:
    """PostgreSQL connector for OSCAR"""
    
    def __init__(self, instance_name: str):
        if instance_name not in DB_INSTANCES:
            raise ValueError(f"Unknown instance: {instance_name}")
        
        self.instance_name = instance_name
        self.config = DB_INSTANCES[instance_name]
        self.project = self.config['project']
        self.region = self.config['region']
        self.instance = self.config['instance']
        self.database = self.config['database']
        self.user = self.config['user']
        self.schema = self.config['schema']
        self.instance_connection_name = f"{self.project}:{self.region}:{self.instance}"
        self.connector = None
        self.engine = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        try:
            logger.info(f"Initializing {self.config['description']} connector")
            
            # SSL disable hack
            if os.path.exists(CA_CERT_PATH):
                os.environ['REQUESTS_CA_BUNDLE'] = CA_CERT_PATH
                os.environ['GOOGLE_AUTH_DISABLE_TLS_VERIFY'] = 'True'
                logger.warning(f"SSL verification disabled: {CA_CERT_PATH}")
            else:
                logger.error(f"CA file not found at: {CA_CERT_PATH}")
                raise FileNotFoundError(f"Required CA file not found at {CA_CERT_PATH}")
            
            # Test authentication
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
            )
            logger.info("Authentication successful")
            
            # Initialize connector
            self.connector = Connector()
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=lambda: self.connector.connect(
                    self.instance_connection_name,
                    "pg8000",
                    user=self.user,
                    db=self.database,
                    enable_iam_auth=True,
                    ip_type="PRIVATE"
                ),
                pool_size=2,
                max_overflow=2,
                pool_pre_ping=True,
                pool_recycle=300
            )
            logger.info("Database engine created successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        try:
            if not self.engine:
                logger.error("Database engine not initialized")
                return False
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                
                if row and row[0] == 1:
                    logger.info(f"{self.instance_name.upper()} connection test successful")
                    return True
                else:
                    logger.error(f"{self.instance_name.upper()} connection test failed")
                    return False
                    
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                columns = result.keys()
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert date objects to string
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    rows.append(row_dict)
                
                logger.info(f"OSCAR Query returned {len(rows)} rows")
                if rows:
                    logger.info(f"Sample row: {rows[0]}")
                
                return rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query was: {query}")
            logger.error(f"Params were: {params}")
            raise


class OracleConnector:
    """Oracle DB connector for CoPPER, EDB, and STAR"""
    
    def __init__(self, instance_name: str):
        if instance_name not in ORACLE_INSTANCES:
            raise ValueError(f"Unknown Oracle instance: {instance_name}")
        
        self.instance_name = instance_name
        self.config = ORACLE_INSTANCES[instance_name]
        self.engine = None
        self._initialize_connection()
    
    def get_tns_string_from_ldap_url(self, ldap_url: str) -> Optional[str]:
        try:
            pattern = r"^ldap://(.+)/(.+),(cn=OracleContext.*)$"
            match = re.match(pattern, ldap_url)
            
            if not match:
                return None
            
            ldap_server, db, ora_context = match.groups()
            
            server = Server(ldap_server)
            conn = Connection(server)
            conn.bind()
            
            conn.search(ora_context, f"(cn={db})", attributes=['orclNetDescString'])
            
            if conn.entries:
                tns = conn.entries[0].orclNetDescString.value
                return tns
            else:
                return None
                
        except Exception as e:
            logger.error(f"LDAP lookup failed: {e}")
            return None
    
    def _initialize_connection(self):
        try:
            logger.info(f"Initializing {self.config['description']} connector")
            
            user = self.config['user']
            password = self.config['password']
            url = self.config['url']
            
            tns_string = self.get_tns_string_from_ldap_url(url)
            
            if tns_string:
                self.engine = sqlalchemy.create_engine(
                    f"oracle+oracledb://{user}:{password}@{tns_string}"
                )
                logger.info(f"{self.instance_name.upper()} engine created successfully")
            else:
                raise Exception("Failed to retrieve TNS string from LDAP")
                
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} initialization failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        try:
            if not self.engine:
                logger.error("Database engine not initialized")
                return False
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT CURRENT_DATE FROM DUAL"))
                row = result.fetchone()
                
                if row:
                    logger.info(f"{self.instance_name.upper()} connection test successful")
                    return True
                else:
                    logger.error(f"{self.instance_name.upper()} connection test failed")
                    return False
                    
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                columns = result.keys()
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert date objects to string
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    rows.append(row_dict)
                
                logger.info(f"{self.instance_name.upper()} Query returned {len(rows)} rows")
                if rows:
                    logger.info(f"Sample row: {rows[0]}")
                
                return rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query was: {query}")
            logger.error(f"Params were: {params}")
            raise


class ReconciliationEngine:
    """Main reconciliation logic"""
    
    def __init__(self):
        self.oscar_conn = None
        self.copper_conn = None
        self.edb_conn = None
        self.star_conn = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        try:
            self.oscar_conn = DatabaseConnector('oscar')
            self.copper_conn = OracleConnector('copper')
            self.edb_conn = OracleConnector('edb')
            self.star_conn = OracleConnector('star')
            
            # Test all connections
            if not all([
                self.oscar_conn.test_connection(),
                self.copper_conn.test_connection(),
                self.edb_conn.test_connection(),
                self.star_conn.test_connection()
            ]):
                raise Exception("One or more database connections failed")
                
            logger.info("All database connections established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def get_oscar_data_by_guid(self, guid: str) -> List[Dict]:
        """Get OSCAR data by GUID - FIXED VERSION"""
        query = """
        SELECT
            GUID,
            (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText/text()', guses.XML))[1]::text as gfid,
            (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as gus_id,
            (xpath('//globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as exchange,
            namespace,
            CASE
                WHEN (xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM dv01cosrs.active_xml_data_store guses
        WHERE guses.GUID = :guid
        AND (guses.namespace = 'GlobexUserSignature')
        LIMIT 1
        """
        
        logger.info(f"Executing OSCAR query with GUID: {guid}")
        results = self.oscar_conn.execute_query(query, {'guid': guid})
        logger.info(f"OSCAR results: {results}")
        return results
    
    def get_oscar_data_by_gfid_gusid(self, gfid: str, gus_id: str) -> List[Dict]:
        """Get OSCAR data by GFID and GUS_ID - FIXED VERSION"""
        query = """
        SELECT
            guid,
            (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText/text()', guses.XML))[1]::text as gfid,
            (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as gus_id,
            (xpath('//globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as exchange,
            namespace,
            CASE
                WHEN (xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM dv01cosrs.active_xml_data_store guses
        WHERE (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText/text()', guses.XML))[1]::text = :gfid
        AND (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text = :gus_id
        AND namespace = 'GlobexUserSignature'
        LIMIT 1
        """
        
        logger.info(f"Executing OSCAR query with GFID: {gfid}, GUS_ID: {gus_id}")
        results = self.oscar_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
        logger.info(f"OSCAR results: {results}")
        return results
    
    def get_copper_data_by_guid(self, guid: str) -> List[Dict]:
        query = """
        SELECT
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            CASE
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM QF44CCRDO.trd_gpid
        WHERE guid = :guid
        """
        
        logger.info(f"Executing CoPPER query with GUID: {guid}")
        results = self.copper_conn.execute_query(query, {'guid': guid})
        logger.info(f"CoPPER results: {results}")
        return results
    
    def get_copper_data_by_gfid_gusid(self, gfid: str, gus_id: str) -> List[Dict]:
        query = """
        SELECT
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            exch_id,
            CASE
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM QF44CCRDO.trd_gpid
        WHERE gfid_id = :gfid AND gpid_id = :gus_id
        """
        
        logger.info(f"Executing CoPPER query with GFID: {gfid}, GUS_ID: {gus_id}")
        results = self.copper_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
        logger.info(f"CoPPER results: {results}")
        return results
    
    def get_edb_data(self, gfid: str, gus_id: str) -> List[Dict]:
        query = """
        SELECT
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            CASE
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM DV02CEDBO.trd_gpid
        WHERE gfid_id = :gfid AND gpid_id = :gus_id
        """
        
        logger.info(f"Executing EDB query with GFID: {gfid}, GUS_ID: {gus_id}")
        results = self.edb_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
        logger.info(f"EDB results: {results}")
        return results
    
    def get_star_data(self, gfid: str) -> List[Dict]:
        query = """
        SELECT gfid, GFID_DESCRIPTION, exchange_id
        FROM ebs_gfid
        WHERE gfid = :gfid
        
        UNION ALL
        
        SELECT gfid, GFID_DESCRIPTION, gfid_type as exchange_id
        FROM bmc_gfid
        WHERE gfid = :gfid
        """
        
        logger.info(f"Executing STAR query with GFID: {gfid}")
        results = self.star_conn.execute_query(query, {'gfid': gfid})
        logger.info(f"STAR results: {results}")
        return results
    
    def determine_scenario(self, oscar_status: str, copper_status: str, 
                          oscar_data: Dict, copper_data: Dict) -> Tuple[str, str]:
        """Determine which scenario applies and return scenario name and recommended action"""
        
        # Check if data exists
        oscar_exists = bool(oscar_data)
        copper_exists = bool(copper_data)
        
        if not oscar_exists and not copper_exists:
            return "No Data", "No records found in either OSCAR or CoPPER"
        
        if not copper_exists:
            if oscar_status == 'ACTIVE':
                return "Scenario 2.4", "OSCAR Active GUID is not present in CoPPER - Check why it is missing and take appropriate action"
            elif oscar_status == 'INACTIVE':
                return "Scenario 2.3", "OSCAR Expired GUID is missing in CoPPER - Run Sync FLAG to N Job"
        
        if not oscar_exists and copper_exists:
            return "Data Mismatch", "Record exists in CoPPER but not in OSCAR - Investigate data inconsistency"
        
        # Both exist - compare statuses
        if oscar_status == 'INACTIVE' and copper_status == 'INACTIVE':
            # Check if GFID and GUS_ID match
            if oscar_data.get('gfid') == copper_data.get('gfid') and \
               oscar_data.get('gus_id') == copper_data.get('gus_id'):
                return "Scenario 2.1", "Both Expired with same GFID and GUS ID - Run MASS SYNC job and then run Sync FLAG to N Job"
            else:
                return "Scenario 2.1 (Mismatch)", "Both Expired but GFID/GUS_ID mismatch - Verify data integrity before sync"
        
        if oscar_status == 'INACTIVE' and copper_status == 'ACTIVE':
            # Check if present with different GUID
            if oscar_data.get('gfid') != copper_data.get('gfid') or \
               oscar_data.get('gus_id') != copper_data.get('gus_id'):
                return "Scenario 2.2", "OSCAR expired, CoPPER active with different GUID and GFID/GUS combination - Run Sync FLAG to N Job"
            else:
                return "Scenario 2.2 (Variant)", "OSCAR expired but CoPPER active - Run Sync FLAG to N Job"
        
        if oscar_status == 'ACTIVE' and copper_status == 'ACTIVE':
            # Verify all fields match
            if oscar_data.get('guid') == copper_data.get('guid') and \
               oscar_data.get('gfid') == copper_data.get('gfid') and \
               oscar_data.get('gus_id') == copper_data.get('gus_id'):
                return "Match", "All fields match - No action required"
            else:
                return "Mismatch", "Active in both but data mismatch - Verify and reconcile data"
        
        return "Unknown Scenario", "Unable to determine scenario - Manual review required"
    
    def perform_reconciliation(self, guid: str = None, gfid: str = None, 
                              gus_id: str = None) -> Dict:
        """Main reconciliation function"""
        
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'search_criteria': {},
                'oscar_copper_edb': {},
                'oscar_copper_star': {},
                'summary': {},
                'scenario': '',
                'recommended_action': ''
            }
            
            # Determine search mode
            if guid:
                result['search_criteria']['type'] = 'guid'
                result['search_criteria']['guid'] = guid
                
                # Get OSCAR data
                oscar_data = self.get_oscar_data_by_guid(guid)
                oscar_record = oscar_data[0] if oscar_data else {}
                
                # Get CoPPER data
                copper_data = self.get_copper_data_by_guid(guid)
                copper_record = copper_data[0] if copper_data else {}
                
            elif gfid and gus_id:
                result['search_criteria']['type'] = 'gfid_gusid'
                result['search_criteria']['gfid'] = gfid
                result['search_criteria']['gus_id'] = gus_id
                
                # Get OSCAR data
                oscar_data = self.get_oscar_data_by_gfid_gusid(gfid, gus_id)
                oscar_record = oscar_data[0] if oscar_data else {}
                
                # Get CoPPER data
                copper_data = self.get_copper_data_by_gfid_gusid(gfid, gus_id)
                copper_record = copper_data[0] if copper_data else {}
                
            else:
                return {
                    'error': 'Invalid search criteria. Provide either GUID or both GFID and GUS_ID',
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"OSCAR Record: {oscar_record}")
            logger.info(f"CoPPER Record: {copper_record}")
            
            # Determine scenario
            oscar_status = oscar_record.get('status', 'MISSING')
            copper_status = copper_record.get('status', 'MISSING')
            scenario, action = self.determine_scenario(
                oscar_status, copper_status, oscar_record, copper_record
            )
            
            result['scenario'] = scenario
            result['recommended_action'] = action
            
            # Get GFID and GUS_ID for EDB and STAR queries
            # Prefer CoPPER data if available
            query_gfid = copper_record.get('gfid') or oscar_record.get('gfid')
            query_gus_id = copper_record.get('gus_id') or oscar_record.get('gus_id')
            
            # OSCAR-CoPPER-EDB comparison
            edb_data = []
            if query_gfid and query_gus_id:
                edb_data = self.get_edb_data(query_gfid, query_gus_id)
            
            edb_record = edb_data[0] if edb_data else {}
            
            result['oscar_copper_edb'] = {
                'oscar': oscar_record,
                'copper': copper_record,
                'edb': edb_record,
                'comparison': {
                    'oscar_copper_match': self._compare_records(oscar_record, copper_record),
                    'copper_edb_match': self._compare_records(copper_record, edb_record),
                    'oscar_status': oscar_status,
                    'copper_status': copper_status,
                    'edb_status': edb_record.get('status', 'MISSING')
                }
            }
            
            # OSCAR-CoPPER-STAR comparison
            star_data = []
            if query_gfid:
                star_data = self.get_star_data(query_gfid)
            
            star_record = star_data[0] if star_data else {}
            
            result['oscar_copper_star'] = {
                'oscar': oscar_record,
                'copper': copper_record,
                'star': star_record,
                'comparison': {
                    'oscar_copper_match': self._compare_records(oscar_record, copper_record),
                    'copper_gfid_in_star': bool(star_record),
                    'oscar_status': oscar_status,
                    'copper_status': copper_status,
                    'star_exists': bool(star_record)
                }
            }
            
            # Calculate summary
            result['summary'] = self._calculate_summary(result)
            
            logger.info(f"Reconciliation result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            }
    
    def _compare_records(self, record1: Dict, record2: Dict) -> str:
        """Compare two records and return match status"""
        if not record1 and not record2:
            return 'BOTH_MISSING'
        if not record1:
            return 'FIRST_MISSING'
        if not record2:
            return 'SECOND_MISSING'
        
        # Compare key fields
        key_fields = ['guid', 'gfid', 'gus_id']
        matches = []
        
        for field in key_fields:
            val1 = record1.get(field)
            val2 = record2.get(field)
            if val1 and val2:
                matches.append(str(val1).strip().upper() == str(val2).strip().upper())
        
        if all(matches) and len(matches) > 0:
            return 'MATCH'
        elif any(matches):
            return 'PARTIAL_MATCH'
        else:
            return 'MISMATCH'
    
    def _calculate_summary(self, result: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {
            'total_comparisons': 2,
            'matches': 0,
            'mismatches': 0,
            'missing': 0
        }
        
        # Check OSCAR-CoPPER-EDB
        edb_comparison = result['oscar_copper_edb']['comparison']
        if edb_comparison['oscar_copper_match'] == 'MATCH' and \
           edb_comparison['copper_edb_match'] == 'MATCH':
            summary['matches'] += 1
        elif 'MISSING' in edb_comparison['oscar_copper_match'] or \
             'MISSING' in edb_comparison['copper_edb_match']:
            summary['missing'] += 1
        else:
            summary['mismatches'] += 1
        
        # Check OSCAR-CoPPER-STAR
        star_comparison = result['oscar_copper_star']['comparison']
        if star_comparison['oscar_copper_match'] == 'MATCH' and \
           star_comparison['copper_gfid_in_star']:
            summary['matches'] += 1
        elif 'MISSING' in star_comparison['oscar_copper_match'] or \
             not star_comparison['copper_gfid_in_star']:
            summary['missing'] += 1
        else:
            summary['mismatches'] += 1
        
        return summary


# Initialize reconciliation engine
recon_engine = None

def get_recon_engine():
    global recon_engine
    if recon_engine is None:
        recon_engine = ReconciliationEngine()
    return recon_engine


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/reconcile', methods=['POST'])
def reconcile():
    try:
        data = request.get_json()
        
        guid = data.get('guid', '').strip()
        gfid = data.get('gfid', '').strip()
        gus_id = data.get('gus_id', '').strip()
        
        logger.info(f"Reconciliation request: guid={guid}, gfid={gfid}, gus_id={gus_id}")
        
        # Validate input
        if not guid and not (gfid and gus_id):
            return jsonify({
                'error': 'Please provide either GUID or both GFID and GUS_ID'
            }), 400
        
        # Get reconciliation engine
        engine = get_recon_engine()
        
        # Perform reconciliation
        result = engine.perform_reconciliation(
            guid=guid if guid else None,
            gfid=gfid if gfid else None,
            gus_id=gus_id if gus_id else None
        )
        
        if 'error' in result:
            logger.error(f"Reconciliation error: {result}")
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        engine = get_recon_engine()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'databases': {
                'oscar': 'connected',
                'copper': 'connected',
                'edb': 'connected',
                'star': 'connected'
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("OSCAR RECONCILIATION TOOL - STARTING")
    print("=" * 60)
    
    # Check if certificate file exists
    import os
    cert_path = CA_CERT_PATH
    if not os.path.exists(cert_path):
        print(f"❌ ERROR: Certificate file not found at: {cert_path}")
        print(f"   Please update CA_CERT_PATH in app.py to point to your new.pem file")
        print(f"   Current working directory: {os.getcwd()}")
        exit(1)
    else:
        print(f"✅ Certificate file found: {cert_path}")
    
    print("\n" + "=" * 60)
    print("Initializing database connections...")
    print("=" * 60)
    
    try:
        # Try to initialize connections on startup
        print("\n🔄 Testing database connections...")
        engine = get_recon_engine()
        print("✅ All database connections initialized successfully!\n")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize connections on startup")
        print(f"   Error: {str(e)}")
        print(f"   The app will try to connect when first request is made.\n")
    
    print("=" * 60)
    print("🚀 Starting Flask application...")
    print("=" * 60)
    print("\n📍 Access the application at: http://localhost:5000")
    print("📍 Health check endpoint: http://localhost:5000/api/health")
    print("\n⏹️  Press CTRL+C to stop the server\n")
    print("=" * 60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start Flask app")
        print(f"   {str(e)}")
        exit(1)
```

## 🔍 Key Fixes in Backend:

### 1. **XPath Column Names Fixed**
The main issue was likely in the XPath extraction. I changed:
```python
# BEFORE (incorrect - was using capital letters inconsistently)
(xpath('...')[1]::text as "GFID"

# AFTER (correct - lowercase aliases)
(xpath('...')[1]::text as gfid
```

### 2. **Added Comprehensive Logging**
Every query now logs:
- What it's searching for
- The full query
- The parameters
- The results returned
- Sample rows

### 3. **Explicit Column Mapping in Queries**
Made sure all queries explicitly name columns:
```python
# OSCAR query now explicitly returns:
# - GUID (uppercase from DB)
# - gfid (lowercase alias)
# - gus_id (lowercase alias)
# - exchange (lowercase alias)
# - namespace
# - status
```

### 4. **Removed Mixed Case Issues**
All field names in results are now lowercase consistently:
- `gfid` (not `GFID` or mixed)
- `gus_id` (not `GUS_ID`)
- `guid` (not `GUID`)

## 🧪 Testing Steps:

1. **Run the app:**
```bash
python3 app.py
```

2. **Check the logs when you search:**
   - You should see: `OSCAR Query returned X rows`
   - You should see: `Sample row: {'guid': '...', 'gfid': '...', 'gus_id': '...'}`

3. **If OSCAR still returns only GUID:**
   - Check the logs carefully
   - The issue might be in the XML structure
   - Try running this query directly in pgAdmin:

```sql
SELECT
    GUID,
    (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText/text()', XML))[1]::text as gfid,
    (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', XML))[1]::text as gus_id,
    namespace
FROM dv01cosrs.active_xml_data_store
WHERE GUID = 'NUEXKEUAEJIN'
AND namespace = 'GlobexUserSignature'
LIMIT 1;
```

4. **Check browser console:**
   - Open Developer Tools (F12)
   - Go to Console tab
   - Look for any JavaScript errors

## 📊 Debug Output to Check:

When you run a search, you should see in terminal:
```
INFO - Reconciliation request: guid=ABCD1234, gfid=, gus_id=
INFO - Executing OSCAR query with GUID: ABCD1234
INFO - OSCAR Query returned 1 rows
INFO - Sample row: {'guid': 'ABCD1234', 'gfid': 'TEST', 'gus_id': 'USR01', 'exchange': 'CME', 'namespace': 'GlobexUserSignature', 'status': 'ACTIVE'}
INFO - OSCAR results: [{'guid': 'ABCD1234', 'gfid': 'TEST', ...}]
INFO - Executing CoPPER query with GUID: ABCD1234
INFO - CoPPER Query returned 1 rows
INFO - Sample row: {'guid': 'ABCD1234', 'gfid': 'TEST', 'gus_id': 'USR01', ...}
```

## 🔧 If Still Having Issues:

### Option A: Test the OSCAR query independently
Create a test script `test_oscar.py`:
```python
from app import DatabaseConnector
import logging

logging.basicConfig(level=logging.INFO)

conn = DatabaseConnector('oscar')
query = """
SELECT
    GUID,
    (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText/text()', XML))[1]::text as gfid,
    (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', XML))[1]::text as gus_id,
    namespace
FROM dv01cosrs.active_xml_data_store
WHERE GUID = :guid
AND namespace = 'GlobexUserSignature'
LIMIT 1
"""

result = conn.execute_query(query, {'guid': 'NUEXKEUAEJIN'})
print("Result:", result)
```

Run it:
```bash
python3 test_oscar.py
```

### Option B: Check XML structure
The XML might have different structure. Add this debug query:
```python
def debug_xml_structure(self, guid: str):
    query = """
    SELECT XML::text
    FROM dv01cosrs.active_xml_data_store
    WHERE GUID = :guid
    LIMIT 1
    """
    results = self.oscar_conn.execute_query(query, {'guid': guid})
    if results:
        print("XML Structure:")
        print(results[0]['xml'])
```

This will show you the actual XML so we can fix the XPath if needed.

### Option C: Simplify the query temporarily
Try this simpler version to isolate the issue:
```python
def get_oscar_data_by_guid(self, guid: str) -> List[Dict]:
    # Simplified query for testing
    query = """
    SELECT
        GUID as guid,
        namespace,
        'test_gfid' as gfid,
        'test_gus' as gus_id,
        'ACTIVE' as status
    FROM dv01cosrs.active_xml_data_store
    WHERE GUID = :guid
    AND namespace = 'GlobexUserSignature'
    LIMIT 1
    """
    return self.oscar_conn.execute_query(query, {'guid': guid})
```

If this works (shows test data), then the issue is definitely in the XPath extraction.

**Let me know what you see in the logs when you run a search!** That will tell us exactly where the problem is. 🔍
