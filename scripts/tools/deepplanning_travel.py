"""
DeepPlanning Travel Tools — ported from DeepPlanning benchmark.

9 domain-specific travel tools that query CSV databases for travel planning.
Each tool is instantiated per-task with the database directory path and language.

Tools:
  1. query_flight_info — Search flights by origin, destination, date
  2. query_train_info — Search trains by origin, destination, date
  3. query_hotel_info — Search hotels with optional filters
  4. recommend_restaurants — Find restaurants near given coordinates
  5. query_restaurant_details — Get details for a specific restaurant
  6. query_attraction_details — Get attraction details by name
  7. recommend_attractions — List attractions in a city
  8. search_location — Get coordinates for a named location
  9. query_road_route_info — Get driving/walking distance and duration
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from .base import Tool

logger = logging.getLogger(__name__)


def _fmt(result: Any) -> str:
    """Format result as JSON string."""
    return json.dumps(result, ensure_ascii=False, indent=2)


def _load_csv(path: str):
    """Load a CSV file as a pandas DataFrame with all string columns."""
    import pandas as pd
    if not os.path.exists(path):
        logger.warning(f"Database file not found: {path}")
        return None
    return pd.read_csv(path, dtype=str)


def _is_nan(v):
    try:
        return v != v
    except Exception:
        return False


def _to_str(v) -> str:
    if v is None or _is_nan(v):
        return ""
    return str(v)


def _to_num(v):
    try:
        if _is_nan(v):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


# ===========================================================================
# Base class for travel tools
# ===========================================================================

class _TravelToolBase(Tool):
    """Base for all travel tools. Holds a pandas DataFrame."""
    output_type = "string"

    def __init__(self, database_path: str, language: str = "en"):
        self._language = language
        self._data = _load_csv(database_path) if database_path else None
        super().__init__()

    def forward(self, **kwargs) -> str:
        raise NotImplementedError


# ===========================================================================
# Flight query
# ===========================================================================

class QueryFlightInfoTool(_TravelToolBase):
    name = "query_flight_info"
    description = (
        "Query flight information by origin city, destination city, departure date, "
        "and optionally seat class. Returns available flights with price and schedule."
    )
    inputs = {
        "origin": {"type": "string", "description": "Origin city name."},
        "destination": {"type": "string", "description": "Destination city name."},
        "depDate": {"type": "string", "description": "Departure date (YYYY-MM-DD)."},
        "seatClassName": {"type": "string", "description": "Optional seat class filter.", "nullable": True},
    }

    def forward(self, origin: str, destination: str, depDate: str, seatClassName: str = "") -> str:
        if self._data is None:
            return "No information found, please check input parameters"

        df = self._data
        q = df[(df["origin_city"] == origin) & (df["destination_city"] == destination) & (df["dep_date"] == depDate)]
        if seatClassName:
            q = q[q["seat_class"] == seatClassName]
        if q.empty:
            return f"No flight information found from {origin} to {destination} on {depDate}"

        flights = []
        for route_idx in sorted(q["route_index"].unique()):
            segs = q[q["route_index"] == route_idx].sort_values("segment_index")
            route_data = {}
            route_price = None
            for idx, row in enumerate(segs.itertuples(), 1):
                seat_status = _to_str(getattr(row, "seat_status", ""))
                if not seat_status or seat_status.lower() == "nan":
                    seat_status = "Available"
                route_data[f"Segment {idx}"] = {
                    "arrCityName": row.destination_city,
                    "arrStationCode": row.arr_station_code,
                    "arrStationName": row.arr_station_name,
                    "depCityName": row.origin_city,
                    "depStationCode": row.dep_station_code,
                    "depStationName": row.dep_station_name,
                    "duration": int(row.duration),
                    "arrDateTime": row.arr_datetime,
                    "depDateTime": row.dep_datetime,
                    "marketingTransportName": row.airline,
                    "marketingTransportNo": row.flight_no,
                    "seatClassName": row.seat_class,
                    "Remaining Seats": seat_status,
                    "equipSize": getattr(row, "equip_size", ""),
                    "equipType": getattr(row, "equip_type", ""),
                    "manufacturer": getattr(row, "manufacturer", ""),
                }
                if idx == 1:
                    try:
                        route_price = float(row.price)
                    except Exception:
                        route_price = None
            route_data["price"] = route_price if route_price is not None else 0
            flights.append(route_data)
        return _fmt(flights)


# ===========================================================================
# Train query
# ===========================================================================

class QueryTrainInfoTool(_TravelToolBase):
    name = "query_train_info"
    description = (
        "Query train information by origin city, destination city, departure date, "
        "and optionally seat class. Returns available trains with price and schedule."
    )
    inputs = {
        "origin": {"type": "string", "description": "Origin city name."},
        "destination": {"type": "string", "description": "Destination city name."},
        "depDate": {"type": "string", "description": "Departure date (YYYY-MM-DD)."},
        "seatClassName": {"type": "string", "description": "Optional seat class filter.", "nullable": True},
    }

    def forward(self, origin: str, destination: str, depDate: str, seatClassName: str = "") -> str:
        if self._data is None:
            return "No information found"

        df = self._data
        q = df[(df["origin_city"] == origin) & (df["destination_city"] == destination) & (df["dep_date"] == depDate)]
        if seatClassName:
            q = q[q["seat_class"] == seatClassName]
        if q.empty:
            return f"No train information found from {origin} to {destination} on {depDate}"

        routes = []
        for route_idx in sorted(q["route_index"].unique()):
            segs = q[q["route_index"] == route_idx].sort_values("segment_index")
            route_data = {}
            route_price = None
            prev_row = None
            for idx, row in enumerate(segs.itertuples(), 1):
                seat_status = _to_str(getattr(row, "seat_status", ""))
                if not seat_status or seat_status.lower() == "nan":
                    seat_status = "Available"
                dep_city = row.origin_city
                if idx > 1 and prev_row:
                    try:
                        dep_city = prev_row.arr_station_name.split(" Station")[0]
                    except Exception:
                        pass
                route_data[f"Segment {idx}"] = {
                    "arrCityName": row.destination_city,
                    "arrStationCode": row.arr_station_code,
                    "arrStationName": row.arr_station_name,
                    "depCityName": dep_city,
                    "depStationCode": row.dep_station_code,
                    "depStationName": row.dep_station_name,
                    "duration": int(row.duration),
                    "arrDateTime": row.arr_datetime,
                    "depDateTime": row.dep_datetime,
                    "marketingTransportName": row.train_type,
                    "marketingTransportNo": row.train_no,
                    "seatClassName": row.seat_class,
                    "Remaining Seats": seat_status,
                }
                if idx == 1:
                    try:
                        route_price = float(row.price)
                    except Exception:
                        route_price = None
                prev_row = row
            route_data["price"] = route_price if route_price is not None else 0
            routes.append([route_data])
        return _fmt(routes)


# ===========================================================================
# Hotel query
# ===========================================================================

class QueryHotelInfoTool(_TravelToolBase):
    name = "query_hotel_info"
    description = (
        "Query hotel information by destination, check-in/check-out dates, "
        "and optional filters for star rating and brand."
    )
    inputs = {
        "destination": {"type": "string", "description": "Destination city name."},
        "checkinDate": {"type": "string", "description": "Check-in date (YYYY-MM-DD)."},
        "checkoutDate": {"type": "string", "description": "Check-out date (YYYY-MM-DD)."},
        "hotelStar": {"type": "string", "description": "Optional hotel star rating filter.", "nullable": True},
        "hotelBrands": {"type": "string", "description": "Optional hotel brand filter.", "nullable": True},
    }

    def forward(self, destination: str, checkinDate: str, checkoutDate: str,
                hotelStar: str = "", hotelBrands: str = "") -> str:
        if self._data is None:
            return "Database not loaded"

        q = self._data
        if hotelStar:
            q = q[q["hotel_star"] == hotelStar]
        if hotelBrands:
            q = q[q["brand"] == hotelBrands]
        if q.empty:
            return f"No hotel information found in {destination} from {checkinDate} to {checkoutDate}"

        results = []
        for _, row in q.iterrows():
            result = {
                "name": _to_str(row.get("name", "")),
                "address": _to_str(row.get("address", "")),
                "latitude": _to_str(row.get("latitude", "")),
                "longitude": _to_str(row.get("longitude", "")),
                "decorationTime": _to_str(row.get("decoration_time", "")),
                "hotelStar": _to_str(row.get("hotel_star", "")),
                "price": _to_str(row.get("price", "")),
                "score": _to_str(row.get("score", "")),
                "brand": _to_str(row.get("brand", "")),
            }
            if "services" in row.index:
                svc = row.get("services", "")
                if svc and isinstance(svc, str) and svc.strip():
                    result["services"] = svc.split(";")
            results.append(result)
        return _fmt(results)


# ===========================================================================
# Restaurant recommendation
# ===========================================================================

class RecommendRestaurantsTool(_TravelToolBase):
    name = "recommend_restaurants"
    description = (
        "Recommend restaurants near a given location. "
        "Requires latitude and longitude coordinates."
    )
    inputs = {
        "latitude": {"type": "string", "description": "Latitude of the location."},
        "longitude": {"type": "string", "description": "Longitude of the location."},
    }

    def forward(self, latitude: str, longitude: str) -> str:
        if self._data is None:
            return "Database not loaded"

        df = self._data
        if "query_latitude" in df.columns and "query_longitude" in df.columns:
            q = df[(df["query_latitude"].astype(str) == str(latitude)) &
                    (df["query_longitude"].astype(str) == str(longitude))]
        else:
            q = df.iloc[0:0]

        if q.empty:
            return f"No recommended restaurants found near coordinates ({latitude}, {longitude})"

        results = []
        for _, row in q.iterrows():
            result = {
                "name": row.get("restaurant_name", ""),
                "latitude": str(row.get("latitude", 0)),
                "longitude": str(row.get("longitude", 0)),
                "price_per_person": str(row.get("price_per_person", 0)),
                "cuisine": row.get("cuisine", ""),
                "opening_time": row.get("opening_time", ""),
                "closing_time": row.get("closing_time", ""),
                "nearby_attraction_name": row.get("nearby_attraction_name", ""),
                "rating": str(row.get("rating", 4.5)),
            }
            if "tags" in row.index:
                tags = row.get("tags", "")
                if tags and isinstance(tags, str) and tags.strip():
                    result["tags"] = tags.split(";")
            results.append(result)
        return _fmt(results)


# ===========================================================================
# Restaurant details
# ===========================================================================

class QueryRestaurantDetailsTool(_TravelToolBase):
    name = "query_restaurant_details"
    description = "Query detailed information about a specific restaurant by name."
    inputs = {
        "restaurant_name": {"type": "string", "description": "Name of the restaurant."},
    }

    def forward(self, restaurant_name: str) -> str:
        if self._data is None:
            return _fmt({"message": "Database not loaded", "restaurant_name": restaurant_name})

        df = self._data
        if "restaurant_name" in df.columns:
            q = df[df["restaurant_name"] == restaurant_name]
        else:
            q = df.iloc[0:0]

        if q.empty:
            return _fmt({"message": f"Detailed information not found for restaurant {restaurant_name}",
                         "restaurant_name": restaurant_name})

        row = q.iloc[0]
        result = {
            "id": row.get("restaurant_id", ""),
            "name": row.get("restaurant_name", restaurant_name),
            "latitude": str(row.get("latitude", 0)),
            "longitude": str(row.get("longitude", 0)),
            "price_per_person": str(row.get("price_per_person", "100")),
            "cuisine": row.get("cuisine", ""),
            "opening_time": row.get("opening_time", ""),
            "closing_time": row.get("closing_time", ""),
            "nearby_attraction_name": row.get("nearby_attraction_name", ""),
            "rating": str(row.get("rating", 4.0)),
        }
        if "tags" in row.index:
            tags = row.get("tags", "")
            if tags and isinstance(tags, str) and tags.strip():
                result["tags"] = tags.split(";")
        return _fmt(result)


# ===========================================================================
# Attraction details
# ===========================================================================

class QueryAttractionDetailsTool(_TravelToolBase):
    name = "query_attraction_details"
    description = "Query detailed information about a specific attraction by name."
    inputs = {
        "attraction_name": {"type": "string", "description": "Name of the attraction."},
    }

    # Language-aware labels matching original DeepPlanning implementation
    _LANG_FIELDS = {
        'zh': {
            'db_not_loaded': "数据库未加载",
            'not_found': lambda name: f"未找到景点 {name} 的详细信息",
            'attraction_id': "景点ID",
            'attraction_name': "景点名称",
            'city': "所属城市",
            'address': "地址",
            'coordinates': "经纬度坐标",
            'latitude': "纬度",
            'longitude': "经度",
            'description': "景点简介",
            'rating': "用户评分",
            'visitor_rating': "（游客平均评价）",
            'opening_hours': "开放时间",
            'to': "至",
            'closed_dates': "闭馆日期",
            'min_visit_hours': "建议最短游玩时长",
            'max_visit_hours': "建议最长游玩时长",
            'hours_unit': "小时",
            'ticket_price': "门票价格",
            'currency_unit': "元",
            'attraction_type': "景点类型",
        },
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda name: f"Detailed information not found for attraction {name}",
            'attraction_id': "Attraction ID",
            'attraction_name': "Attraction Name",
            'city': "City",
            'address': "Address",
            'coordinates': "Coordinates",
            'latitude': "Latitude",
            'longitude': "Longitude",
            'description': "Description",
            'rating': "Rating",
            'visitor_rating': "(average visitor rating)",
            'opening_hours': "Opening Hours",
            'to': "to",
            'closed_dates': "Closed Dates",
            'min_visit_hours': "Minimum Visit Duration",
            'max_visit_hours': "Maximum Visit Duration",
            'hours_unit': "hours",
            'ticket_price': "Ticket Price",
            'currency_unit': "RMB",
            'attraction_type': "Attraction Type",
        },
    }

    def forward(self, attraction_name: str) -> str:
        fields = self._LANG_FIELDS.get(self._language, self._LANG_FIELDS['en'])

        if self._data is None:
            return fields['db_not_loaded']

        df = self._data
        rows = df[df["attraction_name"] == attraction_name]
        if rows.empty:
            return fields['not_found'](attraction_name)

        row = rows.iloc[0]
        rating = _to_num(row.get("rating", None))
        min_h = _to_num(row.get("min_visit_hours", None))
        max_h = _to_num(row.get("max_visit_hours", None))
        ticket = _to_num(row.get("ticket_price", None))

        result = {
            "attraction_id": str(row.get("attraction_id", "")),
            "attraction_name": str(row.get("attraction_name", attraction_name)),
            "city": str(row.get("city", "")),
            "address": str(row.get("address", "")),
            "latitude": str(row.get("latitude", "")),
            "longitude": str(row.get("longitude", "")),
            "description": str(row.get("description", "")),
            "rating": rating if rating is not None else "",
            "opening_time": str(row.get("opening_time", "")),
            "closing_time": str(row.get("closing_time", "")),
            "closing_dates": str(row.get("closing_dates", "")),
            "min_visit_hours": min_h if min_h is not None else "",
            "max_visit_hours": max_h if max_h is not None else "",
            "ticket_price": ticket if ticket is not None else "0",
            "attraction_type": str(row.get("attraction_type", "")),
        }

        # Format as text with language-aware labels (matching original tool output)
        lines = [
            f"{fields['attraction_id']}：{result['attraction_id']}",
            f"{fields['attraction_name']}：{result['attraction_name']}",
            f"{fields['city']}：{result['city']}",
            f"{fields['address']}：{result['address']}",
            f"{fields['coordinates']}：{fields['latitude']} {result['latitude']}，{fields['longitude']} {result['longitude']}",
            f"{fields['description']}：{result['description']}",
            f"{fields['rating']}：{result['rating']}{fields['visitor_rating']}",
        ]
        ot = result["opening_time"]
        ct = result["closing_time"]
        if ot == ct:
            lines.append(f"{fields['opening_hours']}：{ot}")
        else:
            lines.append(f"{fields['opening_hours']}：{ot} {fields['to']} {ct}")
        lines.append(f"{fields['closed_dates']}：{result['closing_dates']}")
        lines.append(f"{fields['min_visit_hours']}：{result['min_visit_hours']} {fields['hours_unit']}")
        lines.append(f"{fields['max_visit_hours']}：{result['max_visit_hours']} {fields['hours_unit']}")
        lines.append(f"{fields['ticket_price']}：{result['ticket_price']} {fields['currency_unit']}")
        lines.append(f"{fields['attraction_type']}：{result['attraction_type']}")
        return "\n".join(lines)


# ===========================================================================
# Attraction recommendation
# ===========================================================================

class RecommendAttractionsTool(_TravelToolBase):
    name = "recommend_attractions"
    description = "Recommend attractions in a city, optionally filtered by type."
    inputs = {
        "city": {"type": "string", "description": "City name."},
        "attraction_type": {"type": "string", "description": "Optional attraction type filter.", "nullable": True},
    }

    def forward(self, city: str, attraction_type: str = "") -> str:
        if self._data is None:
            return "Database not loaded"

        df = self._data
        rows = df  # return all without city filter (matches original)
        if attraction_type:
            rows = rows[rows["attraction_type"] == attraction_type]
        if rows.empty:
            return "No attraction recommendations found"

        lines = ["Recommended attractions:\n"]
        for _, r in rows.iterrows():
            name = r.get("attraction_name", "")
            desc = r.get("description", "")
            atype = r.get("attraction_type", "")
            lines.append(f"{name}, {desc}. This is a {atype} type attraction")
        return "\n".join(lines)


# ===========================================================================
# Location search
# ===========================================================================

class SearchLocationTool(_TravelToolBase):
    name = "search_location"
    description = "Search for coordinates of a named location (hotel, attraction, station, etc.)."
    inputs = {
        "place_name": {"type": "string", "description": "Name of the place to look up."},
    }

    def forward(self, place_name: str) -> str:
        if self._data is None:
            return "Database not loaded"

        df = self._data
        col = "poi_name" if "poi_name" in df.columns else "place_name"
        q = df[df[col] == place_name]
        if q.empty:
            return (
                f"Coordinate information not found for location {place_name}, please check: "
                "1. Whether the place name comes from other tool results; "
                "2. Whether the place name is exactly consistent with tool results"
            )

        row = q.iloc[0]
        result = {
            "place_name": row.get("poi_name", row.get("place_name", place_name)),
            "latitude": str(row.get("latitude", "")),
            "longitude": str(row.get("longitude", "")),
        }
        return _fmt(result)


# ===========================================================================
# Road route query
# ===========================================================================

class QueryRoadRouteInfoTool(_TravelToolBase):
    name = "query_road_route_info"
    description = (
        "Query driving/walking/taxi distance and duration between two locations. "
        "Requires origin and destination as 'latitude,longitude' coordinate strings."
    )
    inputs = {
        "origin": {"type": "string", "description": "Origin coordinates as 'latitude,longitude'."},
        "destination": {"type": "string", "description": "Destination coordinates as 'latitude,longitude'."},
    }

    def forward(self, origin: str, destination: str) -> str:
        if self._data is None:
            return "Database not loaded"

        df = self._data
        # Check coordinate existence
        all_origins = set(df["origin"].unique())
        all_dests = set(df["destination"].unique())
        all_coords = all_origins | all_dests

        if origin not in all_coords:
            return (
                f"Coordinate {origin} is not in query range, please check:\n"
                "1. Whether coordinate comes from valid tool query result;\n"
                "2. Whether coordinate precision is exactly consistent with query result, 6 decimal places"
            )
        if destination not in all_coords:
            return (
                f"Coordinate {destination} is not in query range, please check:\n"
                "1. Whether coordinate comes from valid tool query result;\n"
                "2. Whether coordinate precision is exactly consistent with query result, 6 decimal places"
            )

        q = df[(df["origin"] == origin) & (df["destination"] == destination)]
        if q.empty:
            return f"No transportation information found from {origin} to {destination}"

        row = q.iloc[0]
        result = {
            "origin": row.get("origin", origin),
            "destination": row.get("destination", destination),
            "distance_in_meters": int(row.get("distance_meters", 0)),
            "duration_in_minutes": int(row.get("duration_minutes", 0)),
            "cost": int(row.get("cost", 0)),
        }
        return _fmt(result)


# ===========================================================================
# Factory function
# ===========================================================================

# Database file mapping: tool_name -> CSV subpath relative to database_dir
TRAVEL_DB_MAP = {
    "query_train_info": "trains/trains.csv",
    "query_flight_info": "flights/flights.csv",
    "query_hotel_info": "hotels/hotels.csv",
    "query_attraction_details": "attractions/attractions.csv",
    "recommend_attractions": "attractions/attractions.csv",
    "search_location": "locations/locations_coords.csv",
    "query_road_route_info": "transportation/distance_matrix.csv",
    "recommend_restaurants": "restaurants/restaurants.csv",
    "query_restaurant_details": "restaurants/restaurants.csv",
}

TRAVEL_TOOL_CLASSES = [
    QueryFlightInfoTool,
    QueryTrainInfoTool,
    QueryHotelInfoTool,
    RecommendRestaurantsTool,
    QueryRestaurantDetailsTool,
    QueryAttractionDetailsTool,
    RecommendAttractionsTool,
    SearchLocationTool,
    QueryRoadRouteInfoTool,
]


def create_travel_tools(database_dir: str, language: str = "en") -> Dict[str, Tool]:
    """Create all 9 travel tools bound to a specific database directory.

    Args:
        database_dir: Path to the database directory containing CSV subdirectories
                     (flights/, trains/, hotels/, restaurants/, attractions/, etc.).
        language: Language code ('en' or 'zh').

    Returns:
        Dict mapping tool name -> Tool instance.
    """
    tools = {}
    for cls in TRAVEL_TOOL_CLASSES:
        db_subpath = TRAVEL_DB_MAP.get(cls.name, "")
        db_path = os.path.join(database_dir, db_subpath) if db_subpath else ""
        tool = cls(database_path=db_path, language=language)
        tools[tool.name] = tool
    return tools
