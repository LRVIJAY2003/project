apiVersion: v1
kind: ConfigMap
metadata:
  name: product-endpoints
  namespace: app-2644-default
  labels:
    app.kubernetes.io/instance: prober-checkout
    app.kubernetes.io/name: prober-checkout
    app_id: "2644"
    component_id: coml-5046
    department_name: referential_data_services
    environment: dv
data:
  endpoints.json: |
    {
      "Product-instrument-cache": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/instrumentsByIndexKey",
        "method": "POST",
        "payload": {
          "indexKeys": [
            "Instrument:vmInstrId:42344358",
            "Instrument:vmInstrId:42203334"
          ]
        }
      },
      "Product-BloombergTicker": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/bloombergTickers/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": []
        }
      },
      "Product-instrument": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/instruments/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "string",
              "value": "Y",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-instrumentticks": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/instrumentticks/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "prodTyc",
              "value": "FWD",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-instrumentRelationships": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/instrumentRelationships/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "FXS",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-products": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/products/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "EBS",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-calendar": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/calendars/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "clsName"
          ]
        }
      },
      "Product-options": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/options/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "optExerMnr"
          ]
        }
      },
      "Product-valueDates": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/valueDates/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "EBS",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-productrelationships": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/productrelationships/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "prodTyc"
          ]
        }
      },
      "Product-marketControls": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/marketControls/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "kclpfMstat"
          ]
        }
      },
      "Product-dailyInterestRates": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/dailyInterestRates/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "ticRefixInd"
          ]
        }
      },
      "Product-dailyFXRates": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/dailyFXRates/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "curoId"
          ]
        }
      },
      "Product-deliveryBaskets": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/deliveryBaskets/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": []
        }
      },
      "Product-repoAccruedInterests": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/repoAccruedInterests/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "pmtInstGuid",
              "value": "22",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "Product-GC-Basket": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/gcBasket/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "pmtInstGuid",
              "value": "22",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "Product-Coupon-Scheduler": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/couponSchedules/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": []
        }
      },
      "Product-Instrument-Creation-Service": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/instrumentcreationservice/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "productName",
              "value": "TEST CORN OPTION",
              "operation": "EQ"
            }
          ]
        }
      },
      "Product-UPI-UAT": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/upi/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "as"
          ]
        }
      },
      "Product-Options-Series-Ticks": {
        "host": "coml-5046-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/optionseriesticks/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "NYMEX",
              "operation": "EQ"
            }
          ]
        }
      }
    }

















apiVersion: v1
kind: ConfigMap
metadata:
  name: entity-endpoints
  namespace: app-2644-default
  labels:
    app.kubernetes.io/instance: prober-checkout
    app.kubernetes.io/name: prober-checkout
    app_id: "2644"
    component_id: coml-5200
    department_name: referential_data_services
    environment: dv
data:
  endpoints.json: |
    {
      "entity-cloBillingInvoiceGroup": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/cloBillingInvoiceGroup/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "bilGrp"
          ]
        }
      },
      "entity-Session": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/sessions/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "glcInd"
          ]
        }
      },
      "entity-sessionEBSPerm": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/sessionEBSPerm/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "usrLocSym",
              "value": "C",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "entity-sessionBTECPerm": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/sessionBTECPerm/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "usrVnuAbbr",
              "value": "US",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "entity-uniPortalCode": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/uniPortalCode/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "uniCd"
          ]
        }
      },
      "entity-cmiFirmEntitlement": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/cmiFirmEntitlement/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "mfer",
              "value": "EBS",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "entity-clearingRelationship": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/clearingRelationship/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "clrOrgId"
          ]
        }
      },
      "entity-Organization": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/organizations/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "orgTy"
          ]
        }
      },
      "entity-GUS": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/globalUserSignature/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "EBS",
              "operation": "CONTAINS"
            }
          ]
        }
      },
      "entity-position": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/positionaccounts/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "glbActRefKey"
          ]
        }
      },
      "entity-firm": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/firm/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "extType"
          ]
        }
      },
      "entity-serviceGroup": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/serviceGroups/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "grp"
          ]
        }
      },
      "entity-Farm-F&O": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/fnoMarketsFirms/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "firmType"
          ]
        }
      },
      "entity-GUSPREM": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/globalUserSignaturePerm/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [
            {
              "tag": "exchId",
              "value": "BTUS",
              "operation": "EQ"
            }
          ]
        }
      },
      "entity-Exchange": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/exchange/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "tradeGrouping"
          ]
        }
      },
      "entity-country": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/country/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": [],
          "distinct": "true",
          "fields": [
            "ctryName"
          ]
        }
      },
      "entity-user": {
        "host": "coml-5200-app-2644-default.ant-usc1.prj-dv-anthos-host.dev.gcp.cme.com",
        "path": "/api/instance1/user/page=0&size=10",
        "method": "POST",
        "payload": {
          "req": []
        }
      }
    }















apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-prober
  namespace: app-2644-default
  labels:
    app.kubernetes.io/name: product-prober
    app_id: "2644"
    component_id: coml-5046
    department_name: referential_data_services
    environment: dv
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: product-prober
  template:
    metadata:
      labels:
        app.kubernetes.io/name: product-prober
    spec:
      containers:
      - name: prober
        image: your-registry/prober-image:latest
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        env:
        - name: CONFIG_PATH
          value: "/etc/config/endpoints.json"
      volumes:
      - name: config-volume
        configMap:
          name: product-endpoints














apiVersion: apps/v1
kind: Deployment
metadata:
  name: entity-prober
  namespace: app-2644-default
  labels:
    app.kubernetes.io/name: entity-prober
    app_id: "2644"
    component_id: coml-5200
    department_name: referential_data_services
    environment: dv
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: entity-prober
  template:
    metadata:
      labels:
        app.kubernetes.io/name: entity-prober
    spec:
      containers:
      - name: prober
        image: your-registry/prober-image:latest
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        env:
        - name: CONFIG_PATH
          value: "/etc/config/endpoints.json"
      volumes:
      - name: config-volume
        configMap:
          name: entity-endpoints