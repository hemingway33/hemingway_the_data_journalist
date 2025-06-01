"""
Credit scoring API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import structlog

from ...core.database import get_db
from ...risk_engine.credit_scoring import CreditScoringEngine
from ...models.risk_factors import CreditEntity, CreditRiskScore
from ..schemas import (
    CreditScoringRequest, CreditScore, EntityFeatures,
    CreditEntity as CreditEntitySchema, CreditEntityCreate, CreditEntityUpdate,
    MessageResponse
)

logger = structlog.get_logger()

router = APIRouter()


@router.post("/score", response_model=CreditScore)
async def score_entity(
    request: CreditScoringRequest,
    db: Session = Depends(get_db)
):
    """
    Score a credit entity using the risk factor store.
    """
    try:
        # Initialize scoring engine
        scoring_engine = CreditScoringEngine(db)
        
        # Convert Pydantic model to dict
        entity_features = request.entity_features.dict()
        
        # Perform scoring
        score_result = scoring_engine.score_entity(
            entity_id=request.entity_id,
            entity_features=entity_features,
            as_of_date=request.as_of_date
        )
        
        # Store score in database
        entity = db.query(CreditEntity).filter(
            CreditEntity.entity_id == request.entity_id
        ).first()
        
        if entity:
            credit_score = CreditRiskScore(
                entity_id=entity.id,
                score_date=score_result['score_date'],
                probability_of_default=score_result['probability_of_default'],
                loss_given_default=score_result['loss_given_default'],
                exposure_at_default=score_result['exposure_at_default'],
                expected_loss=score_result['expected_loss'],
                model_version=score_result['model_version'],
                model_type=score_result['model_type'],
                confidence_interval_lower=score_result.get('confidence_interval_lower'),
                confidence_interval_upper=score_result.get('confidence_interval_upper'),
                risk_factors_snapshot=score_result['risk_factors_snapshot']
            )
            db.add(credit_score)
            db.commit()
        
        return CreditScore(**score_result)
        
    except Exception as e:
        logger.error("Credit scoring failed", entity_id=request.entity_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Credit scoring failed: {str(e)}")


@router.get("/entities", response_model=List[CreditEntitySchema])
async def list_entities(
    skip: int = 0,
    limit: int = 100,
    entity_type: str = None,
    industry_code: str = None,
    db: Session = Depends(get_db)
):
    """
    List credit entities.
    """
    query = db.query(CreditEntity).filter(CreditEntity.is_active == True)
    
    if entity_type:
        query = query.filter(CreditEntity.entity_type == entity_type)
    
    if industry_code:
        query = query.filter(CreditEntity.industry_code == industry_code)
    
    entities = query.offset(skip).limit(limit).all()
    return entities


@router.post("/entities", response_model=CreditEntitySchema)
async def create_entity(
    entity: CreditEntityCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new credit entity.
    """
    # Check if entity already exists
    existing = db.query(CreditEntity).filter(
        CreditEntity.entity_id == entity.entity_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Entity already exists")
    
    db_entity = CreditEntity(**entity.dict())
    db.add(db_entity)
    db.commit()
    db.refresh(db_entity)
    
    return db_entity


@router.get("/entities/{entity_id}", response_model=CreditEntitySchema)
async def get_entity(entity_id: str, db: Session = Depends(get_db)):
    """
    Get a specific credit entity.
    """
    entity = db.query(CreditEntity).filter(
        CreditEntity.entity_id == entity_id,
        CreditEntity.is_active == True
    ).first()
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity


@router.put("/entities/{entity_id}", response_model=CreditEntitySchema)
async def update_entity(
    entity_id: str,
    entity_update: CreditEntityUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a credit entity.
    """
    entity = db.query(CreditEntity).filter(
        CreditEntity.entity_id == entity_id
    ).first()
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Update fields
    for field, value in entity_update.dict(exclude_unset=True).items():
        setattr(entity, field, value)
    
    entity.updated_at = datetime.now()
    db.commit()
    db.refresh(entity)
    
    return entity


@router.get("/entities/{entity_id}/scores", response_model=List[CreditScore])
async def get_entity_scores(
    entity_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get credit scores for an entity.
    """
    entity = db.query(CreditEntity).filter(
        CreditEntity.entity_id == entity_id
    ).first()
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    scores = db.query(CreditRiskScore).filter(
        CreditRiskScore.entity_id == entity.id
    ).order_by(CreditRiskScore.score_date.desc()).limit(limit).all()
    
    # Convert to schema format
    score_list = []
    for score in scores:
        score_dict = {
            'entity_id': entity.entity_id,
            'score_date': score.score_date,
            'probability_of_default': score.probability_of_default,
            'loss_given_default': score.loss_given_default or 0.45,
            'exposure_at_default': score.exposure_at_default or 0,
            'expected_loss': score.expected_loss or 0,
            'risk_rating': 'BBB',  # Would calculate from PD
            'model_version': score.model_version,
            'model_type': score.model_type or 'xgboost',
            'confidence_interval_lower': score.confidence_interval_lower,
            'confidence_interval_upper': score.confidence_interval_upper,
            'risk_factors_snapshot': score.risk_factors_snapshot or {}
        }
        score_list.append(CreditScore(**score_dict))
    
    return score_list


@router.post("/batch-score")
async def batch_score_entities(
    entity_ids: List[str],
    entity_features: EntityFeatures,
    background_tasks: BackgroundTasks,
    as_of_date: datetime = None,
    db: Session = Depends(get_db)
):
    """
    Score multiple entities in batch (background processing).
    """
    def process_batch():
        scoring_engine = CreditScoringEngine(db)
        features_dict = entity_features.dict()
        
        for entity_id in entity_ids:
            try:
                score_result = scoring_engine.score_entity(
                    entity_id=entity_id,
                    entity_features=features_dict,
                    as_of_date=as_of_date
                )
                
                # Store in database
                entity = db.query(CreditEntity).filter(
                    CreditEntity.entity_id == entity_id
                ).first()
                
                if entity:
                    credit_score = CreditRiskScore(
                        entity_id=entity.id,
                        score_date=score_result['score_date'],
                        probability_of_default=score_result['probability_of_default'],
                        loss_given_default=score_result['loss_given_default'],
                        exposure_at_default=score_result['exposure_at_default'],
                        expected_loss=score_result['expected_loss'],
                        model_version=score_result['model_version'],
                        model_type=score_result['model_type'],
                        confidence_interval_lower=score_result.get('confidence_interval_lower'),
                        confidence_interval_upper=score_result.get('confidence_interval_upper'),
                        risk_factors_snapshot=score_result['risk_factors_snapshot']
                    )
                    db.add(credit_score)
                
                logger.info("Batch scoring completed", entity_id=entity_id)
                
            except Exception as e:
                logger.error("Batch scoring failed", entity_id=entity_id, error=str(e))
        
        db.commit()
    
    background_tasks.add_task(process_batch)
    
    return MessageResponse(
        message=f"Batch scoring started for {len(entity_ids)} entities",
        success=True
    ) 