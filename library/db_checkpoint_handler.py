"""
Database checkpoint handler - saves checkpoint metadata to database

This module provides direct DB updates during the training loop, replacing the old
file-watching approach in watchdog_server.py.

Key features:
- Saves checkpoint metadata immediately after checkpoint file is written
- Raises exceptions on errors, causing training to halt if DB save fails
- Supports both regular model training (Dreambooth) and LoRA training
- Handles SD3.5 (base_model='1') and Flux1 (base_model='3') models

The function save_checkpoint_to_db() is called from:
- sd3_train_utils.py: save_models() - for SD3.5 model training
- flux_train_utils.py: save_models() - for Flux1 model training  
- train_network.py: save_model() - for LoRA training (both SD3 and Flux)

Error handling:
- FileNotFoundError: Checkpoint file doesn't exist
- ValueError: Checkpoint in wrong directory or model not found in DB
- RuntimeError/Exception: DB operation fails (insert/update)

All errors will propagate up and stop the training process, ensuring data consistency.
"""
import os
import sys
import logging
import traceback
from typing import Optional

# Add parent directory to path to import Flask app
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logger
logger = logging.getLogger(__name__)


def save_checkpoint_to_db(
    checkpoint_path: str,
    model_id: str,
    step_num: int,
    is_lora: bool = False,
    base_model: str = '1'  # '1' for SD3.5, '3' for Flux1
):
    """
    Save checkpoint metadata to database
    
    Args:
        checkpoint_path: Full path to the checkpoint file
        model_id: The training model/client ID
        step_num: The training step number
        is_lora: Whether this is a LoRA model
        base_model: Base model type ('1' for SD3.5, '3' for Flux1)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is not in expected directory or model not found
        RuntimeError: If database operation fails
    """
    logger.info(f"[DB Checkpoint Handler] Starting checkpoint save to DB - Model ID: {model_id}, Step: {step_num}, Is LoRA: {is_lora}, Base Model: {base_model}")
    logger.debug(f"[DB Checkpoint Handler] Checkpoint path: {checkpoint_path}")
    
    # Import here to avoid circular imports and ensure Flask context
    from app import app as flask_app
    from core.models.db_models import TempModel, TempLora
    from os import environ
    
    output_base_path = environ.get('OUTPUT_LEARNING_PICTURES_BASE_PATH', '/deployment/outputs/train')
    model_dir = os.path.join(output_base_path, str(model_id))
    logger.debug(f"[DB Checkpoint Handler] Model directory: {model_dir}")
    
    # Verify the file exists and is in the correct directory
    logger.debug(f"[DB Checkpoint Handler] Verifying checkpoint file exists: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"[DB Checkpoint Handler] Checkpoint file does not exist: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    logger.info(f"[DB Checkpoint Handler] ✓ Checkpoint file exists: {checkpoint_path}")
        
    logger.debug(f"[DB Checkpoint Handler] Verifying checkpoint is in correct directory")
    if not checkpoint_path.startswith(model_dir):
        logger.error(f"[DB Checkpoint Handler] Checkpoint not in expected directory: {checkpoint_path} (expected: {model_dir})")
        raise ValueError(f"Checkpoint not in expected directory: {checkpoint_path} (expected to start with {model_dir})")
    logger.info(f"[DB Checkpoint Handler] ✓ Checkpoint is in correct directory")
        
    filename = os.path.basename(checkpoint_path)
    logger.debug(f"[DB Checkpoint Handler] Checkpoint filename: {filename}")
    
    # Find icon path
    icon_path = os.path.join(model_dir, "icon.png")
    logger.debug(f"[DB Checkpoint Handler] Checking for icon at: {icon_path}")
    if not os.path.exists(icon_path):
        logger.warning(f"[DB Checkpoint Handler] Icon not found at {icon_path} - continuing without icon")
        icon_path = None
    else:
        logger.debug(f"[DB Checkpoint Handler] ✓ Icon found at: {icon_path}")
        
    # Create checkpoint name
    checkpoint_name = f"Checkpoint {step_num}"
    logger.debug(f"[DB Checkpoint Handler] Checkpoint name: {checkpoint_name}")
    
    # Extract clip file paths for SD3.5
    base_filename_parts = filename.split('.')
    base_filename = base_filename_parts[0]
    extension = base_filename_parts[1] if len(base_filename_parts) > 1 else "safetensors"
    
    clip_g_path = os.path.join(model_dir, f"{base_filename}_clip_g.{extension}")
    clip_l_path = os.path.join(model_dir, f"{base_filename}_clip_l.{extension}")
    logger.debug(f"[DB Checkpoint Handler] Clip paths - G: {clip_g_path}, L: {clip_l_path}")
    
    # Use Flask app context
    logger.debug(f"[DB Checkpoint Handler] Entering Flask app context for DB operations")
    try:
        with flask_app.app_context():
            if is_lora:
                # Handle LoRA checkpoint
                logger.info(f"[DB Checkpoint Handler] Processing LoRA checkpoint for model ID: {model_id}")
                logger.debug(f"[DB Checkpoint Handler] Querying TempLora table for ID={model_id}, Name='TempLora'")
                try:
                    lora_obj = TempLora.select_where(ID=model_id, Name="TempLora").first()
                except Exception as e:
                    logger.error(f"[DB Checkpoint Handler] ❌ ERROR querying TempLora table for ID={model_id}, Name='TempLora'")
                    logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                    logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                    logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Failed to query LoRA model from database: {str(e)}") from e
                
                if not lora_obj:
                    logger.error(f"[DB Checkpoint Handler] LoRA model {model_id} not found in database")
                    raise ValueError(f"LoRA model {model_id} not found in database")
                logger.info(f"[DB Checkpoint Handler] ✓ Found LoRA model in DB - ID: {lora_obj.ID}, BaseModel: {lora_obj.BaseModel}")
                    
                # Upsert into TempLora (composite PK: ID + Name)
                logger.debug(f"[DB Checkpoint Handler] Checking if checkpoint '{checkpoint_name}' already exists in TempLora")
                try:
                    existing = TempLora.select_where(ID=lora_obj.ID, Name=checkpoint_name).first()
                except Exception as e:
                    logger.error(f"[DB Checkpoint Handler] ❌ ERROR querying TempLora for existing checkpoint - ID={lora_obj.ID}, Name='{checkpoint_name}'")
                    logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                    logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                    logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Failed to check for existing LoRA checkpoint in database: {str(e)}") from e
                
                if existing is not None:
                    logger.info(f"[DB Checkpoint Handler] Checkpoint '{checkpoint_name}' exists - updating record")
                    logger.debug(f"[DB Checkpoint Handler] Update data - Path: {checkpoint_path}, IconPath: {icon_path}")
                    try:
                        existing.update(
                            Path=checkpoint_path,
                            BaseModel=lora_obj.BaseModel,
                            ModelId=lora_obj.ModelId,
                            Type='User-Trained',
                            IconPath=icon_path,
                        )
                        logger.info(f"[DB Checkpoint Handler] ✓ Successfully updated LoRA checkpoint '{checkpoint_name}' in database")
                    except Exception as e:
                        logger.error(f"[DB Checkpoint Handler] ❌ ERROR updating LoRA checkpoint '{checkpoint_name}' in database")
                        logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                        logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                        logger.error(f"[DB Checkpoint Handler] Update data that failed: Path={checkpoint_path}, BaseModel={lora_obj.BaseModel}, ModelId={lora_obj.ModelId}")
                        logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Failed to update LoRA checkpoint in database: {str(e)}") from e
                else:
                    logger.info(f"[DB Checkpoint Handler] Checkpoint '{checkpoint_name}' does not exist - inserting new record")
                    logger.debug(f"[DB Checkpoint Handler] Insert data - ID: {lora_obj.ID}, Name: {checkpoint_name}, Path: {checkpoint_path}")
                    try:
                        TempLora.insert_new(
                            ID=lora_obj.ID,
                            Name=checkpoint_name,
                            Path=checkpoint_path,
                            BaseModel=lora_obj.BaseModel,
                            ModelId=lora_obj.ModelId,
                            Type='User-Trained',
                            IconPath=icon_path,
                        )
                        logger.info(f"[DB Checkpoint Handler] ✓ Successfully inserted LoRA checkpoint '{checkpoint_name}' into database")
                    except Exception as e:
                        logger.error(f"[DB Checkpoint Handler] ❌ ERROR inserting LoRA checkpoint '{checkpoint_name}' into database")
                        logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                        logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                        logger.error(f"[DB Checkpoint Handler] Insert data that failed: ID={lora_obj.ID}, Name={checkpoint_name}, Path={checkpoint_path}")
                        logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Failed to insert LoRA checkpoint into database: {str(e)}") from e
            else:
                # Handle regular model checkpoint
                logger.info(f"[DB Checkpoint Handler] Processing regular model checkpoint for model ID: {model_id}")
                logger.debug(f"[DB Checkpoint Handler] Querying TempModel table for ID={model_id}, ModelName='TempModel'")
                try:
                    model_obj = TempModel.select_where(ID=model_id, ModelName="TempModel").first()
                except Exception as e:
                    logger.error(f"[DB Checkpoint Handler] ❌ ERROR querying TempModel table for ID={model_id}, ModelName='TempModel'")
                    logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                    logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                    logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Failed to query model from database: {str(e)}") from e
                
                if not model_obj:
                    logger.error(f"[DB Checkpoint Handler] Model {model_id} not found in database")
                    raise ValueError(f"Model {model_id} not found in database")
                logger.info(f"[DB Checkpoint Handler] ✓ Found model in DB - ID: {model_obj.ID}, BaseModel: {model_obj.BaseModel}")
                    
                # Get default paths for models that need them
                t5_path = model_obj.T5Path
                ae_path = model_obj.AePath
                logger.debug(f"[DB Checkpoint Handler] Model paths - T5: {t5_path}, AE: {ae_path}")
                
                # Upsert into TempModel (composite PK: ID + ModelName)
                logger.debug(f"[DB Checkpoint Handler] Checking if checkpoint '{checkpoint_name}' already exists in TempModel")
                try:
                    existing = TempModel.select_where(ID=model_obj.ID, ModelName=checkpoint_name).first()
                except Exception as e:
                    logger.error(f"[DB Checkpoint Handler] ❌ ERROR querying TempModel for existing checkpoint - ID={model_obj.ID}, ModelName='{checkpoint_name}'")
                    logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                    logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                    logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Failed to check for existing checkpoint in database: {str(e)}") from e
                
                if existing is not None:
                    logger.info(f"[DB Checkpoint Handler] Checkpoint '{checkpoint_name}' exists - updating record")
                    logger.debug(f"[DB Checkpoint Handler] Update data - CkptPath: {checkpoint_path}, BaseModel: {base_model}")
                    try:
                        existing.update(
                            CkptPath=checkpoint_path,
                            ClipGPath=clip_g_path if base_model == '1' else None,
                            ClipLPath=clip_l_path if base_model == '1' else model_obj.ClipLPath,
                            T5Path=t5_path,
                            AePath=ae_path if base_model == '3' else None,
                            BaseModel=base_model,
                            ModelType='User-Trained',
                            IconPath=icon_path,
                        )
                        logger.info(f"[DB Checkpoint Handler] ✓ Successfully updated checkpoint '{checkpoint_name}' in database")
                    except Exception as e:
                        logger.error(f"[DB Checkpoint Handler] ❌ ERROR updating checkpoint '{checkpoint_name}' in database")
                        logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                        logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                        logger.error(f"[DB Checkpoint Handler] Update data that failed: CkptPath={checkpoint_path}, BaseModel={base_model}, ClipGPath={clip_g_path if base_model == '1' else None}")
                        logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Failed to update checkpoint in database: {str(e)}") from e
                else:
                    logger.info(f"[DB Checkpoint Handler] Checkpoint '{checkpoint_name}' does not exist - inserting new record")
                    logger.debug(f"[DB Checkpoint Handler] Insert data - ID: {model_obj.ID}, ModelName: {checkpoint_name}, CkptPath: {checkpoint_path}")
                    try:
                        TempModel.insert_new(
                            ID=model_obj.ID,
                            ModelName=checkpoint_name,
                            CkptPath=checkpoint_path,
                            ClipGPath=clip_g_path if base_model == '1' else None,
                            ClipLPath=clip_l_path if base_model == '1' else model_obj.ClipLPath,
                            T5Path=t5_path,
                            AePath=ae_path if base_model == '3' else None,
                            BaseModel=base_model,
                            ModelType='User-Trained',
                            IconPath=icon_path
                        )
                        logger.info(f"[DB Checkpoint Handler] ✓ Successfully inserted checkpoint '{checkpoint_name}' into database")
                    except Exception as e:
                        logger.error(f"[DB Checkpoint Handler] ❌ ERROR inserting checkpoint '{checkpoint_name}' into database")
                        logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
                        logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
                        logger.error(f"[DB Checkpoint Handler] Insert data that failed: ID={model_obj.ID}, ModelName={checkpoint_name}, CkptPath={checkpoint_path}")
                        logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Failed to insert checkpoint into database: {str(e)}") from e
    except (ValueError, FileNotFoundError):
        # Re-raise validation errors as-is (they already have proper logging)
        raise
    except Exception as e:
        # Catch any other unexpected errors and log them
        logger.error(f"[DB Checkpoint Handler] ❌ UNEXPECTED ERROR during checkpoint save operation")
        logger.error(f"[DB Checkpoint Handler] Error type: {type(e).__name__}")
        logger.error(f"[DB Checkpoint Handler] Error message: {str(e)}")
        logger.error(f"[DB Checkpoint Handler] Model ID: {model_id}, Step: {step_num}, Is LoRA: {is_lora}")
        logger.error(f"[DB Checkpoint Handler] Full traceback:\n{traceback.format_exc()}")
        raise
    
    logger.info(f"[DB Checkpoint Handler] ✓ Checkpoint save completed successfully - Model ID: {model_id}, Step: {step_num}")


def extract_step_number_from_filename(filename: str, save_interval: Optional[int] = None) -> int:
    """
    Extract step number from checkpoint filename
    
    Args:
        filename: The checkpoint filename (e.g., "sd3-000010.safetensors")
        save_interval: The save interval used during training
        
    Returns:
        The step number
    """
    logger.debug(f"[DB Checkpoint Handler] Extracting step number from filename: {filename}")
    
    if '-' in filename:
        # Extract the step number (e.g., 000010 from sd3-000010.safetensors)
        step_part = filename.split('-')[1].split('.')[0]
        step_num = int(step_part)
        logger.debug(f"[DB Checkpoint Handler] Extracted step number from filename: {step_num}")
        return step_num
    elif save_interval:
        # If no step number in filename, estimate based on save interval
        logger.debug(f"[DB Checkpoint Handler] No step number in filename, using save_interval: {save_interval}")
        return save_interval
    else:
        logger.warning(f"[DB Checkpoint Handler] Could not extract step number from filename '{filename}' and no save_interval provided, returning 0")
        return 0

