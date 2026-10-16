---
name: mongodb-repository-layer
description: Step 2 of FinRAG 2.0 implementation - MongoDB repository layer with tenant isolation
metadata:
  type: project
---

## Step 2: MongoDB Repository Layer + Tenant Isolation

### Context
Building on Step 1 foundations (structured logging, error handling, OmniRoute configuration), we now need to implement Step 2: replacing the SQLAlchemy/SQLite database layer with MongoDB using the Motor async driver, implementing a repository layer with tenant isolation, and migrating core models.

This step is critical for FinRAG 2.0 as specified in the implementation brief, requiring:
- MongoDB replaces SQLAlchemy/SQLite for all relational data
- Repository layer enforces tenant isolation 
- All repository methods require TenantContext and inject tenant_id into queries
- Compound indexes start with tenant_id for performance
- Cross-tenant isolation tests must fail closed

### Implementation Plan

#### 1. Update Dependencies
- Remove SQLAlchemy, Alembic dependencies
- Add Motor (async MongoDB driver) and PyMongo
- Update pyproject.toml dependencies

#### 2. Create Repository Layer
- BaseRepository class with tenant isolation enforcement
- TenantContext class to carry tenant/user info
- Repository methods that require tenant context
- Automatic tenant_id injection in all queries
- Compound indexing strategy starting with tenant_id

#### 3. Migrate Core Models
Convert SQLAlchemy models to MongoDB documents:
- Tenant → tenants collection
- User → users collection  
- Workspace → workspaces collection
- Document → documents collection
- Chunk → chunks collection
- StructuredFinancial → structured_financials collection
- Conversation → conversations collection
- Message → messages collection
- Job → jobs collection
- EvalRun/EvalItem → eval_runs/eval_items collections
- FeedbackLog → feedback_logs collection

#### 4. Implement Migration Script
- One-time import from SQLite schema (if exists)
- Data transformation from relational to document model
- Preserve existing data relationships

#### 5. Build Isolation Tests
- Cross-tenant access tests that must fail
- Attempt to read/update/delete/search another tenant's data
- Verify all access is properly blocked
- Test every repository method and API route

### Files to Modify
- pyproject.toml - Update dependencies
- app/db/ - Create new MongoDB-based implementation
- app/core/config.py - Add MongoDB configuration
- Repository implementations for each model type
- Test files to verify isolation

### Verification
After implementation, all existing tests should pass (adapted for MongoDB) and new tenant isolation tests should verify proper access control.

### Dependencies on Previous Steps
- Uses structured logging from Step 1
- Uses enhanced error handling from Step 1
- Uses OmniRoute configuration from Step 1

### Estimated Effort
This step involves significant data layer changes and should be implemented incrementally with testing at each stage.