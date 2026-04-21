# Run this to patch knowledge_base.py build() method
import re

patch = '''    def build(self):
        """Build memvid memory file + hybrid indexes from pending chunks."""
        if not self._pending:
            print("No pending chunks to build.")
            return

        print(f"\\nBuilding knowledge base \'{self.name}\'...")
        print(f"Total chunks: {len(self._pending)}")

        # Add each chunk to memvid encoder individually
        # Use large chunk_size so memvid does not re-split our pre-chunked text
        for chunk in self._pending:
            self.encoder.add_text(
                chunk[\'text\'],
                chunk_size=10000,   # larger than any chunk we produce
                overlap=0,
            )

        # Build memvid .mp4 memory file
        print("Building memvid memory file...")
        self.encoder.build_video(self.video_path, self.index_path)

        # Build hybrid retriever on top of same chunks
        print("Building hybrid retriever...")
        self.retriever = HybridRetriever(self.index_path + \'_hybrid\')
        self.retriever.build(self._pending)
        self.retriever.save(self.index_path + \'_hybrid\')

        # Save chunks for future reference
        with open(self.chunks_path, \'wb\') as f:
            pickle.dump(self._pending, f)

        # Update and save stats
        self._stats[\'total_chunks\'] = len(self._pending)
        with open(self.stats_path, \'w\') as f:
            json.dump(self._stats, f, indent=2)

        print(f"\\nKnowledge base built successfully!")
        print(f"  Memory file : {self.video_path}")
        print(f"  Index       : {self.index_path}")
        print(f"  Chunks      : {len(self._pending)}")
        print(f"  Languages   : {self.target_langs}")

        self._pending = []
'''
print("Patch ready — apply manually or run fix_kb.py")
