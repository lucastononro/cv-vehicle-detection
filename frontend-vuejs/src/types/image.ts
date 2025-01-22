export interface Image {
  filename: string;
  size: number;
  last_modified: string;
  thumbnail?: string; // Optional base64 encoded thumbnail
} 