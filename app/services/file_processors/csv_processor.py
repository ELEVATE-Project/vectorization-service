import io
import uuid
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from .base_processor import BaseFileProcessor
from app.config import settings
from app.services.translation_service import process_chunk

logger = logging.getLogger(__name__)

# CSV Column name constants
COL_SL_NO = "SL NO"
COL_SUB_CATEGORY = "Sub-cateogry"
COL_TITLE = "TITLE OF THE PROJECT"
COL_TARGET_STAKEHOLDER = "TARGET STAKEHOLDER"
COL_DURATION = "DURATION"
COL_DESCRIPTION = "DESCRIPTION"
COL_OBJECTIVE = "OBJECTIVE"
COL_PROJECT_RESOURCE = "PROJECT LEVEL LEARNING RESOURCE"
COL_TASK_NAME = "TASK NAME"
COL_SUB_TASK = "SUB TASK (If any)"
COL_TASK_RESOURCE = "NAME OF TASK LEVEL LEARNING RESOURCE"


class CSVProcessor(BaseFileProcessor):
    def get_supported_extensions(self) -> List[str]:
        return ['.csv']

    def _extract_main_info(self, sl_group: pd.DataFrame) -> Dict[str, str]:
        """Extract main task information from a group"""
        return {
            "sl_no": str(sl_group[COL_SL_NO].iloc[0]),
            "sub_category": str(sl_group[COL_SUB_CATEGORY].iloc[0]) if COL_SUB_CATEGORY in sl_group else "",
            "project_title": str(sl_group[COL_TITLE].iloc[0]) if COL_TITLE in sl_group else "",
            "target_stakeholder": str(sl_group[COL_TARGET_STAKEHOLDER].iloc[0]) if COL_TARGET_STAKEHOLDER in sl_group else "",
            "duration": str(sl_group[COL_DURATION].iloc[0]) if COL_DURATION in sl_group else "",
            "description": str(sl_group[COL_DESCRIPTION].iloc[0]) if COL_DESCRIPTION in sl_group else "",
            "objective": str(sl_group[COL_OBJECTIVE].iloc[0]) if COL_OBJECTIVE in sl_group else "",
            "project_learning_resource": str(sl_group[COL_PROJECT_RESOURCE].iloc[0]) if COL_PROJECT_RESOURCE in sl_group else "",
        }

    def _extract_tasks(self, sl_group: pd.DataFrame) -> List[Dict[str, str]]:
        """Extract tasks and subtasks from a group"""
        tasks = []
        for _, row in sl_group.iterrows():
            if pd.notna(row.get(COL_TASK_NAME, "")):
                task_info = {
                    "task_name": str(row[COL_TASK_NAME]),
                    "sub_task": str(row[COL_SUB_TASK]) if pd.notna(row.get(COL_SUB_TASK, "")) else "",
                    "task_learning_resource": str(row[COL_TASK_RESOURCE]) if pd.notna(row.get(COL_TASK_RESOURCE, "")) else "",
                }
                tasks.append(task_info)
        return tasks

    def _generate_text_content(self, main_info: Dict[str, str], tasks: List[Dict[str, str]]) -> str:
        """Generate text content for embedding"""
        text_content = f"""
        Task Number: {main_info['sl_no']}
        Category: {main_info['sub_category']}
        Project Title: {main_info['project_title']}
        Target Stakeholder: {main_info['target_stakeholder']}
        Duration: {main_info['duration']}
        Description: {main_info['description']}
        Objective: {main_info['objective']}
        Project Learning Resource: {main_info['project_learning_resource']}

        Tasks and Subtasks:
        """

        for task in tasks:
            text_content += f"\nTask: {task['task_name']}"
            if task["sub_task"]:
                text_content += f"\n  - Subtask: {task['sub_task']}"
            if task["task_learning_resource"]:
                text_content += f"\n  - Learning Resource: {task['task_learning_resource']}"

        return text_content.strip()

    def _create_chunks(self, doc: LangchainDocument, text_splitter, text_content: str) -> List[Dict[str, Any]]:
        """Create processed chunks from a document"""
        processed_chunks = []
        
        if len(text_content) > settings.CHUNK_SIZE:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk_id = uuid.uuid4().hex
                processed_text, is_hindi = process_chunk(chunk.page_content, chunk_id)
                processed_chunk = {
                    "id": chunk_id,
                    "text": processed_text,
                    "metadata": {**chunk.metadata, "is_hindi": is_hindi, "total_chunks": len(chunks)},
                }
                processed_chunks.append(processed_chunk)
        else:
            chunk_id = uuid.uuid4().hex
            processed_text, is_hindi = process_chunk(doc.page_content, chunk_id)
            processed_chunk = {
                "id": chunk_id,
                "text": processed_text,
                "metadata": {**doc.metadata, "is_hindi": is_hindi, "total_chunks": 1},
            }
            processed_chunks.append(processed_chunk)
        
        return processed_chunks

    async def process(self, file_content: bytes, filename: str, priority: str) -> List[Dict[str, Any]]:
        """Process CSV file and return chunks"""
        self._validate_file_content(file_content, filename)

        try:
            df = pd.read_csv(io.BytesIO(file_content))
            logger.info(f"Original columns: {df.columns.tolist()}")

            df.columns = df.columns.str.strip()
            unnamed_cols = [col for col in df.columns if "Unnamed:" in str(col)]
            df = df.drop(columns=unnamed_cols)

            expected_columns = [
                COL_SL_NO, COL_SUB_CATEGORY, COL_TITLE, COL_TARGET_STAKEHOLDER,
                COL_DURATION, COL_DESCRIPTION, COL_OBJECTIVE, COL_PROJECT_RESOURCE,
                COL_TASK_NAME, COL_SUB_TASK, COL_TASK_RESOURCE,
            ]

            missing_columns = [col for col in expected_columns if col not in df.columns]
            if COL_SL_NO in missing_columns:
                raise HTTPException(status_code=400, detail="Missing required column: SL NO")

            main_columns = [
                COL_SL_NO, COL_SUB_CATEGORY, COL_TITLE, COL_TARGET_STAKEHOLDER,
                COL_DURATION, COL_DESCRIPTION, COL_OBJECTIVE, COL_PROJECT_RESOURCE,
            ]
            existing_main_columns = [col for col in main_columns if col in df.columns]
            df[existing_main_columns] = df[existing_main_columns].ffill()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )

            processed_chunks = []

            for sl_no in df[COL_SL_NO].dropna().unique():
                sl_group = df[df[COL_SL_NO] == sl_no].copy()
                main_info = self._extract_main_info(sl_group)
                tasks = self._extract_tasks(sl_group)
                text_content = self._generate_text_content(main_info, tasks)

                doc = LangchainDocument(
                    page_content=text_content,
                    metadata={
                        **main_info,
                        "tasks": tasks,
                        "priority": priority,
                        "source": filename,
                        "type": "project_task",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    },
                )

                processed_chunks.extend(self._create_chunks(doc, text_splitter, text_content))

            logger.info(f"Successfully processed {len(processed_chunks)} chunks from CSV")
            return processed_chunks

        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
