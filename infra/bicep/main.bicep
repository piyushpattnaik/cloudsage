// CloudSage — Azure Bicep Infrastructure
// FIXED:
//   1. Function App has SystemAssigned managed identity (DefaultAzureCredential requires this)
//   2. SERVICE_BUS_CONNECTION_STRING uses listKeys() — no more 'placeholder'
//   3. RBAC role assignments for managed identity on Cosmos, OpenAI, AI Search
//   4. Static Web App repositoryUrl is now a parameter (not hardcoded)
//   5. Application Insights ConnectionString added alongside InstrumentationKey

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment (dev/staging/production)')
param environment string = 'production'

@description('Unique suffix for globally unique resource names')
param suffix string = uniqueString(resourceGroup().id)

@description('GitHub repository URL for the dashboard Static Web App')
param repositoryUrl string = 'https://github.com/your-org/cloudsage'

var prefix = 'cloudsage'
var tags = {
  Application: 'CloudSage'
  Environment: environment
  ManagedBy: 'Bicep'
}

// ─── Well-known built-in role definition GUIDs ────────────────────────────────
var cognitiveServicesUserRoleId      = 'a97b65f3-24c7-4388-baec-2e87135dc908'
var searchIndexDataContributorRoleId = '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
var searchServiceContributorRoleId   = '7ca78c08-252a-4471-8644-bb5ff32d4ba0'

// ─── Event Hub ────────────────────────────────────────────────────────────────
resource eventHubNamespace 'Microsoft.EventHub/namespaces@2023-01-01-preview' = {
  name: '${prefix}-eh-${suffix}'
  location: location
  tags: tags
  sku: { name: 'Standard', tier: 'Standard', capacity: 2 }
  properties: { isAutoInflateEnabled: true, maximumThroughputUnits: 10 }
}

resource eventHub 'Microsoft.EventHub/namespaces/eventhubs@2023-01-01-preview' = {
  parent: eventHubNamespace
  name: 'cloudsage-events'
  properties: { messageRetentionInDays: 7, partitionCount: 4 }
}

// ─── Service Bus ─────────────────────────────────────────────────────────────
resource serviceBusNamespace 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: '${prefix}-sb-${suffix}'
  location: location
  tags: tags
  sku: { name: 'Standard', tier: 'Standard' }
}

resource serviceBusQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBusNamespace
  name: 'cloudsage-actions'
  properties: {
    lockDuration: 'PT5M'
    maxDeliveryCount: 3
    deadLetteringOnMessageExpiration: true
    defaultMessageTimeToLive: 'P1D'
  }
}

// Shared access policy so the Function App can Listen and Send
resource serviceBusAuthRule 'Microsoft.ServiceBus/namespaces/authorizationRules@2022-10-01-preview' = {
  parent: serviceBusNamespace
  name: 'CloudSageFunctionApp'
  properties: { rights: ['Listen', 'Send'] }
}

// ─── Cosmos DB ────────────────────────────────────────────────────────────────
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-09-15' = {
  name: '${prefix}-cosmos-${suffix}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    locations: [{ locationName: location, failoverPriority: 0, isZoneRedundant: false }]
    enableAutomaticFailover: false
    capabilities: [{ name: 'EnableServerless' }]
  }
}

resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-09-15' = {
  parent: cosmosAccount
  name: 'cloudsage'
  properties: { resource: { id: 'cloudsage' } }
}

resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-09-15' = {
  parent: cosmosDatabase
  name: 'incidents'
  properties: {
    resource: {
      id: 'incidents'
      partitionKey: { paths: ['/service'], kind: 'Hash' }
      defaultTtl: 7776000
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/"_etag"/?' }]
      }
    }
  }
}

// ─── Azure OpenAI ─────────────────────────────────────────────────────────────
resource openAIAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${prefix}-openai-${suffix}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: '${prefix}-openai-${suffix}'
  }
}

// ─── Azure AI Search ──────────────────────────────────────────────────────────
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${prefix}-search-${suffix}'
  location: location
  tags: tags
  sku: { name: 'standard' }
  properties: { replicaCount: 1, partitionCount: 1, publicNetworkAccess: 'enabled' }
}

// ─── Log Analytics + Application Insights ────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-law-${suffix}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 90 }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-ai-${suffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    RetentionInDays: 90
  }
}

// ─── Storage + Functions ──────────────────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${prefix}sa${suffix}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}

resource functionsPlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: '${prefix}-func-plan-${suffix}'
  location: location
  tags: tags
  sku: { name: 'Y1', tier: 'Dynamic' }
  properties: { reserved: true }
}

resource functionApp 'Microsoft.Web/sites@2022-09-01' = {
  name: '${prefix}-func-${suffix}'
  location: location
  tags: tags
  kind: 'functionapp,linux'
  // FIXED: SystemAssigned identity — required for DefaultAzureCredential
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: functionsPlan.id
    siteConfig: {
      pythonVersion: '3.11'
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'FUNCTIONS_WORKER_RUNTIME',    value: 'python' }
        // Application Insights — both keys for full SDK compatibility
        { name: 'APPINSIGHTS_INSTRUMENTATIONKEY',         value: appInsights.properties.InstrumentationKey }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING',  value: appInsights.properties.ConnectionString }
        // Azure service endpoints — no secrets needed; managed identity provides auth
        { name: 'COSMOS_DB_ENDPOINT',      value: cosmosAccount.properties.documentEndpoint }
        { name: 'AZURE_OPENAI_ENDPOINT',   value: openAIAccount.properties.endpoint }
        { name: 'AZURE_SEARCH_ENDPOINT',   value: 'https://${searchService.name}.search.windows.net' }
        // FIXED: real Service Bus connection string via listKeys() — replaces 'placeholder'
        { name: 'SERVICE_BUS_CONNECTION_STRING', value: serviceBusAuthRule.listKeys().primaryConnectionString }
        { name: 'ENVIRONMENT', value: environment }
      ]
    }
  }
}

// ─── RBAC for Function App Managed Identity ───────────────────────────────────
// FIXED: without these, DefaultAzureCredential succeeds but all API calls
// return 403 because the identity has no permissions.

// Cosmos DB Built-in Data Contributor (Cosmos-specific RBAC, not ARM)
resource cosmosRoleAssignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2023-09-15' = {
  parent: cosmosAccount
  name: guid(cosmosAccount.id, functionApp.id, 'cosmos-data-contributor')
  properties: {
    // Role ID 00000000-0000-0000-0000-000000000002 = Cosmos DB Built-in Data Contributor
    roleDefinitionId: '${cosmosAccount.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    principalId: functionApp.identity.principalId
    scope: cosmosAccount.id
  }
}

// Cognitive Services User — call Azure OpenAI
resource openAIRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAIAccount.id, functionApp.id, 'cognitive-services-user')
  scope: openAIAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Search Index Data Contributor — read/write index documents
resource searchIndexRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, functionApp.id, 'search-index-contributor')
  scope: searchService
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchIndexDataContributorRoleId)
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Search Service Contributor — manage index schema
resource searchServiceRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, functionApp.id, 'search-service-contributor')
  scope: searchService
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchServiceContributorRoleId)
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// ─── Static Web App (Dashboard) ───────────────────────────────────────────────
resource staticWebApp 'Microsoft.Web/staticSites@2022-09-01' = {
  name: '${prefix}-dashboard-${suffix}'
  location: location
  tags: tags
  sku: { name: 'Standard', tier: 'Standard' }
  properties: {
    // FIXED: repositoryUrl is now a parameter, not hardcoded
    repositoryUrl: repositoryUrl
    branch: 'main'
    buildProperties: {
      appLocation: '/dashboard'
      outputLocation: 'out'
    }
  }
}

// ─── Outputs ─────────────────────────────────────────────────────────────────
output cosmosEndpoint          string = cosmosAccount.properties.documentEndpoint
output openAIEndpoint          string = openAIAccount.properties.endpoint
output searchEndpoint          string = 'https://${searchService.name}.search.windows.net'
output eventHubNamespaceFQDN   string = eventHubNamespace.properties.serviceBusEndpoint
output functionAppUrl          string = 'https://${functionApp.properties.defaultHostName}'
output functionAppPrincipalId  string = functionApp.identity.principalId
output dashboardUrl            string = 'https://${staticWebApp.properties.defaultHostname}'
output logAnalyticsWorkspaceId string = logAnalytics.properties.customerId
